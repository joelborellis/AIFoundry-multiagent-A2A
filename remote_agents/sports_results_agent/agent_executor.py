import logging
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
    new_text_artifact,
)
from agent import OpenAIWebSearchAgent

logging.basicConfig(level=logging.INFO)
# Reduce httpx logging verbosity to avoid OpenAI tracing noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class OpenAIWebSearchAgentExecutor(AgentExecutor):
    """Streams a single 'current_result' artifact with proper append/finalization."""

    def __init__(self):
        self.agent = OpenAIWebSearchAgent()
        self._initialized = False

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        try:
            # Lazy init
            if not self._initialized:
                await self.agent.initialize()
                self._initialized = True
                logger.info("OpenAI Agent initialized successfully")

            # Resolve user input (fallback to message parts)
            query = context.get_user_input()
            if not query and context.message and context.message.parts:
                texts = [p.text for p in context.message.parts if getattr(p, "text", None)]
                query = "\n".join(texts).strip()

            if not query:
                task = context.current_task or new_task(context.message)
                if not context.current_task:
                    await event_queue.enqueue_event(task)
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        status=TaskStatus(
                            state=TaskState.input_required,
                            message=new_agent_text_message(
                                "Please provide your query (no text was detected).",
                                task.context_id,
                                task.id,
                            ),
                        ),
                        final=True,
                        contextId=task.context_id,
                        taskId=task.id,
                    )
                )
                return

            # Ensure task exists & announce working
            task = context.current_task
            if not task:
                task = new_task(context.message)
                await event_queue.enqueue_event(task)

            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    status=TaskStatus(state=TaskState.working),
                    final=False,
                    contextId=task.context_id,
                    taskId=task.id,
                )
            )

            # Stream loop — create artifact once, then append
            result_artifact_id = None

            async for partial in self.agent.stream(query, task.context_id):
                require_input = partial.get("require_user_input", False)
                is_done = partial.get("is_task_complete", False)
                text_content = (partial.get("content") or "").strip()

                if require_input:
                    await event_queue.enqueue_event(
                        TaskStatusUpdateEvent(
                            status=TaskStatus(
                                state=TaskState.input_required,
                                message=new_agent_text_message(
                                    text_content or "Additional input is required.",
                                    task.context_id,
                                    task.id,
                                ),
                            ),
                            final=True,
                            contextId=task.context_id,
                            taskId=task.id,
                        )
                    )
                    return

                if is_done:
                    final_text = text_content or "Task completed."
                    if result_artifact_id is None:
                        # No prior chunks: create once and close
                        artifact = new_text_artifact(
                            name="current_result",
                            description="Result of request to agent.",
                            text=final_text,
                        )
                        result_artifact_id = artifact.artifact_id  # snake_case
                        await event_queue.enqueue_event(
                            TaskArtifactUpdateEvent(
                                append=False,
                                contextId=task.context_id,
                                taskId=task.id,
                                lastChunk=True,
                                artifact=artifact,  # pass the model
                            )
                        )
                    else:
                        # Append final piece to the existing artifact id
                        artifact = {
                            "artifact_id": result_artifact_id,  # snake_case
                            "name": "current_result",
                            "description": "Result of request to agent.",
                            "parts": [{"kind": "text", "text": final_text}],
                        }
                        await event_queue.enqueue_event(
                            TaskArtifactUpdateEvent(
                                append=True,
                                contextId=task.context_id,
                                taskId=task.id,
                                lastChunk=True,
                                artifact=artifact,
                            )
                        )

                    await event_queue.enqueue_event(
                        TaskStatusUpdateEvent(
                            status=TaskStatus(state=TaskState.completed),
                            final=True,
                            contextId=task.context_id,
                            taskId=task.id,
                        )
                    )
                    return

                # Working updates: history + stream to artifact
                if text_content:
                    # history
                    await event_queue.enqueue_event(
                        TaskStatusUpdateEvent(
                            status=TaskStatus(
                                state=TaskState.working,
                                message=new_agent_text_message(
                                    text_content,
                                    task.context_id,
                                    task.id,
                                ),
                            ),
                            final=False,
                            contextId=task.context_id,
                            taskId=task.id,
                        )
                    )

                    # artifact: create on first chunk, append thereafter
                    if result_artifact_id is None:
                        artifact = new_text_artifact(
                            name="current_result",
                            description="Result of request to agent (streaming).",
                            text=text_content,
                        )
                        result_artifact_id = artifact.artifact_id  # snake_case
                        await event_queue.enqueue_event(
                            TaskArtifactUpdateEvent(
                                append=False,      # creation
                                contextId=task.context_id,
                                taskId=task.id,
                                lastChunk=False,
                                artifact=artifact,
                            )
                        )
                    else:
                        artifact = {
                            "artifact_id": result_artifact_id,  # snake_case
                            "name": "current_result",
                            "description": "Result of request to agent (streaming).",
                            "parts": [{"kind": "text", "text": text_content}],
                        }
                        await event_queue.enqueue_event(
                            TaskArtifactUpdateEvent(
                                append=True,       # append to existing
                                contextId=task.context_id,
                                taskId=task.id,
                                lastChunk=False,
                                artifact=artifact,
                            )
                        )

            # Stream ended without explicit completion
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=new_agent_text_message(
                            "Stream ended unexpectedly without completion.",
                            task.context_id,
                            task.id,
                        ),
                    ),
                    final=True,
                    contextId=task.context_id,
                    taskId=task.id,
                )
            )

        except Exception as e:
            logger.exception("Agent execution failed")
            task = context.current_task
            if not task:
                task = new_task(context.message)
                await event_queue.enqueue_event(task)
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=new_agent_text_message(
                            f"Task failed: {e.__class__.__name__}: {e}",
                            task.context_id,
                            task.id,
                        ),
                    ),
                    final=True,
                    contextId=task.context_id,
                    taskId=task.id,
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if task:
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    status=TaskStatus(state=TaskState.canceled),
                    final=True,
                    contextId=task.context_id,
                    taskId=task.id,
                )
            )
