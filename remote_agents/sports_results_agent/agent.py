import logging
from collections.abc import AsyncIterable
from typing import Any, Optional

from agents import Agent, Runner, WebSearchTool  # OpenAI Agents SDK
from openai.types.responses import ResponseTextDeltaEvent  # <- raw delta type

# Reduce httpx logging verbosity to avoid tracing noise
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class OpenAIWebSearchAgent:
    """Wraps OpenAI Agent with WebSearchTool to handle various tasks."""

    def __init__(self, flush_every: int = 200):
        self.agent: Optional[Agent] = None
        self.flush_every = flush_every  # stream chunk size for UX

    async def initialize(self):
        self.agent = Agent(
            name="Sports Results Agent",
            instructions=(
                "You are a helpful agent that searches the web for sports results. "
                "Give concise scores, winner, and a few notable facts. "
                "When you cite, include the source name in parentheses, e.g. (ESPN), (Reuters)."
            ),
            tools=[WebSearchTool()],
        )
        logger.info("OpenAI Agent initialized successfully")

    async def stream(
        self,
        user_input: str,
        session_id: str | None = None,
    ) -> AsyncIterable[dict[str, Any]]:
        """
        Yields dicts your A2A executor understands:
          - content: str
          - is_task_complete: bool
          - require_user_input: bool
        Optionally:
          - event: "token" | "tool_start" | "tool_result" | "tool_error" | "agent_updated"
          - tool_name: str
          - meta: dict
        """
        if not self.agent:
            yield {
                "is_task_complete": False,
                "require_user_input": True,
                "content": "Agent not initialized. Please call initialize() first.",
            }
            return

        result = Runner.run_streamed(self.agent, input=user_input)

        buffer: list[str] = []
        since_flush = 0

        try:
            async for event in result.stream_events():
                etype = getattr(event, "type", None)

                # 1) RAW LLM DELTAS (Responses API format)
                #    event.type == "raw_response_event" and data is ResponseTextDeltaEvent
                if etype == "raw_response_event" and isinstance(getattr(event, "data", None), ResponseTextDeltaEvent):
                    delta = event.data.delta or ""
                    if delta:
                        buffer.append(delta)
                        since_flush += len(delta)
                        # Optional streaming chunks for better UX
                        if since_flush >= self.flush_every:
                            chunk = "".join(buffer)
                            buffer.clear()
                            since_flush = 0
                            yield {
                                "is_task_complete": False,
                                "require_user_input": False,
                                "event": "token",
                                "content": chunk,
                            }
                    continue

                # 2) HIGHER-LEVEL RUN ITEMS
                #    event.type == "run_item_stream_event"
                if etype == "run_item_stream_event":
                    item = getattr(event, "item", None)
                    itype = getattr(item, "type", None)
                    # tool call started
                    if itype == "tool_call_item":
                        yield {
                            "is_task_complete": False,
                            "require_user_input": False,
                            "event": "tool_start",
                            "tool_name": getattr(item, "tool_name", "unknown_tool"),
                            "content": f"Using tool: {getattr(item, 'tool_name', 'unknown_tool')}…",
                            "meta": {
                                "input": getattr(item, "input", None),
                            },
                        }
                    # tool output arrived
                    elif itype == "tool_call_output_item":
                        # Some SDK versions expose 'output' (structured) and/or 'text'
                        output = getattr(item, "output", None)
                        text = getattr(item, "text", None)
                        yield {
                            "is_task_complete": False,
                            "require_user_input": False,
                            "event": "tool_result",
                            "tool_name": getattr(item, "tool_name", "unknown_tool"),
                            "content": text or "Tool returned results.",
                            "meta": {"output": output},
                        }
                    # final model message (one chunk at item granularity)
                    elif itype == "message_output_item":
                        # If you want, you could flush here as well, but we rely on deltas buffer.
                        pass
                    continue

                # 3) AGENT HANDOFF/UPDATE
                if etype == "agent_updated_stream_event":
                    new_agent = getattr(event, "new_agent", None)
                    if new_agent:
                        yield {
                            "is_task_complete": False,
                            "require_user_input": False,
                            "event": "agent_updated",
                            "content": f"Handoff to agent: {getattr(new_agent, 'name', 'unknown')}",
                        }
                    continue

                # 4) TOOL ERRORS (if surfaced as dedicated events in your SDK version)
                if etype == "tool_error":
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "event": "tool_error",
                        "tool_name": getattr(event, "tool_name", "unknown_tool"),
                        "content": f"Tool error: {getattr(event, 'error', 'unknown error')}",
                    }
                    continue

            # End of stream → flush any remaining buffered text as the final answer
            final_text = "".join(buffer).strip()
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": final_text if final_text else "Done.",
            }

        except Exception as e:
            # Let your executor catch this and mark the task failed
            logger.exception("Streaming failed")
            raise
