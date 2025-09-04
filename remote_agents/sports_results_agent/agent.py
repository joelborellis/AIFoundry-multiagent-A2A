import logging
from collections.abc import AsyncIterable
from typing import Any

from agents import Agent, Runner, WebSearchTool

from dotenv import load_dotenv
from pydantic import BaseModel


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()

# region Response Format


class ResponseFormat(BaseModel):
    """A Response Format model to direct how the model should respond."""

    status: str = "input_required"
    message: str


# endregion

# region Azure AI Agent with MCP


class OpenAIWebSearchAgent:
    """Wraps OpenAI Agent with WebSearchTool to handle various tasks."""

    def __init__(self):
        self.agent = None

    async def initialize(self):
        """Initialize the OpenAI agent with WebSearchTool()."""
        try:

            self.agent = Agent(
                name="Sports Results Agent",
                instructions="You are a helpful agent that searches the web for sports results.",
                tools=[WebSearchTool()],
            )

            logger.info("OpenAI Agent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI Agent: {e}")
            await self.cleanup()
            raise

    async def stream(
        self,
        user_input: str,
        session_id: str = None,
    ) -> AsyncIterable[dict[str, Any]]:
        """Stream responses from the OpenAI Agent.

        Args:
            user_input (str): User input message.
            session_id (str): Unique identifier for the session (optional).

        Yields:
            dict: A dictionary containing the content and task completion status.
        """
        if not self.agent:
            yield {
                "is_task_complete": False,
                "require_user_input": True,
                "content": "Agent not initialized. Please call initialize() first.",
            }
            return

        try:
            # Use the stream_events() method to get async iterable events
            stream_result = Runner.run_streamed(
                self.agent,
                user_input,
            )

            async for event in stream_result.stream_events():
                # Look for ResponseTextDeltaEvent in raw_response_event
                if (
                    hasattr(event, "type")
                    and event.type == "raw_response_event"
                    and hasattr(event, "data")
                ):

                    data = event.data
                    data_type = type(data).__name__

                    # Extract text delta from ResponseTextDeltaEvent
                    if data_type == "ResponseTextDeltaEvent" and hasattr(data, "delta"):
                        delta_text = data.delta
                        if delta_text:  # Only yield if there's actual content
                            yield {
                                "is_task_complete": False,
                                "require_user_input": False,
                                "content": delta_text,
                            }

            # Final completion message
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": "Task completed successfully.",
            }
        except Exception as e:
            # Enhanced error handling for rate limits
            error_message = self._parse_rate_limit_error(e)
            yield {
                "is_task_complete": False,
                "require_user_input": True,
                "content": error_message,
            }

    def _parse_rate_limit_error(self, error: Exception) -> str:
        """
        Parse rate limit errors to provide specific information about the type of limit exceeded.

        Args:
            error: The exception that was raised

        Returns:
            str: A descriptive error message indicating the type of rate limit
        """
        error_str = str(error).lower()
        error_type = type(error).__name__

        # Check for common rate limit indicators
        if "rate limit" in error_str or "quota" in error_str or "429" in error_str:
            # Check for tokens per minute (TPM) rate limit
            if any(
                keyword in error_str
                for keyword in ["token", "tpm", "tokens per minute"]
            ):
                return (
                    f"⚠️ **Rate Limit Exceeded: Tokens Per Minute (TPM)**\n\n"
                    f"The request exceeded the allowed tokens per minute limit. "
                    f"This typically means too many tokens are being processed in a short time period.\n\n"
                    f"**Suggestions:**\n"
                    f"• Wait a moment before retrying\n"
                    f"• Reduce the length of your input\n"
                    f"• Break large requests into smaller chunks\n\n"
                    f"**Technical details:** {error_type}: {str(error)}"
                )

            # Check for requests per minute (RPM) rate limit
            elif any(
                keyword in error_str
                for keyword in [
                    "request",
                    "rpm",
                    "requests per minute",
                    "too many requests",
                ]
            ):
                return (
                    f"⚠️ **Rate Limit Exceeded: Requests Per Minute (RPM)**\n\n"
                    f"The request exceeded the allowed requests per minute limit. "
                    f"This typically means too many API calls are being made in a short time period.\n\n"
                    f"**Suggestions:**\n"
                    f"• Wait a moment before retrying\n"
                    f"• Reduce the frequency of requests\n"
                    f"• Implement request batching if possible\n\n"
                    f"**Technical details:** {error_type}: {str(error)}"
                )

            # Check for Azure OpenAI specific patterns within rate limit context
            elif "azure" in error_str and (
                "openai" in error_str or "cognitive" in error_str
            ):
                return (
                    f"⚠️ **Azure OpenAI Quota Exceeded**\n\n"
                    f"Your Azure OpenAI service quota has been exceeded. This could be either "
                    f"tokens per minute (TPM) or requests per minute (RPM).\n\n"
                    f"**Suggestions:**\n"
                    f"• Check your Azure OpenAI resource usage in the Azure portal\n"
                    f"• Wait for the quota to reset\n"
                    f"• Consider upgrading your Azure OpenAI pricing tier\n\n"
                    f"**Technical details:** {error_type}: {str(error)}"
                )

            # Check for OpenAI specific patterns within rate limit context
            elif "openai" in error_str:
                return (
                    f"⚠️ **OpenAI Quota Exceeded**\n\n"
                    f"Your OpenAI API quota has been exceeded. This could be either "
                    f"tokens per minute (TPM) or requests per minute (RPM).\n\n"
                    f"**Suggestions:**\n"
                    f"• Check your OpenAI API usage dashboard\n"
                    f"• Wait for the quota to reset\n"
                    f"• Consider upgrading your OpenAI plan\n\n"
                    f"**Technical details:** {error_type}: {str(error)}"
                )

            # Generic rate limit error
            else:
                return (
                    f"⚠️ **Rate Limit Exceeded**\n\n"
                    f"A rate limit has been exceeded, but the specific type could not be determined.\n\n"
                    f"**Suggestions:**\n"
                    f"• Wait a moment before retrying\n"
                    f"• Check your API usage and limits\n"
                    f"• Consider reducing request frequency or size\n\n"
                    f"**Technical details:** {error_type}: {str(error)}"
                )

        # Check for usage/limit patterns outside of explicit rate limit context
        elif ("usage" in error_str or "limit" in error_str) and (
            "openai" in error_str or "azure" in error_str
        ):
            if "azure" in error_str:
                return (
                    f"⚠️ **Azure Service Usage Limit Reached**\n\n"
                    f"Your Azure service has reached its usage limit.\n\n"
                    f"**Suggestions:**\n"
                    f"• Check your Azure resource usage in the Azure portal\n"
                    f"• Wait for the limit to reset\n"
                    f"• Consider upgrading your service tier\n\n"
                    f"**Technical details:** {error_type}: {str(error)}"
                )
            else:
                return (
                    f"⚠️ **OpenAI Usage Limit Reached**\n\n"
                    f"Your OpenAI API has reached its usage limit.\n\n"
                    f"**Suggestions:**\n"
                    f"• Check your OpenAI API usage dashboard\n"
                    f"• Wait for the limit to reset\n"
                    f"• Consider upgrading your OpenAI plan\n\n"
                    f"**Technical details:** {error_type}: {str(error)}"
                )

        # Default error handling for non-rate-limit errors
        return f"Error processing request: {str(error)}"

    async def cleanup(self):
        """Cleanup resources."""

        self.agent = None
