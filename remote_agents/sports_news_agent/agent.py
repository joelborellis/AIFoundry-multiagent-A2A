import logging
import os
from collections.abc import AsyncIterable
from typing import Any

from azure.identity.aio import DefaultAzureCredential
from dotenv import load_dotenv
from pydantic import BaseModel
from semantic_kernel.agents import AzureAIAgent
from semantic_kernel.connectors.mcp import MCPStreamableHttpPlugin

logger = logging.getLogger(__name__)

load_dotenv()

# region Response Format


class ResponseFormat(BaseModel):
    """A Response Format model to direct how the model should respond."""

    status: str = 'input_required'
    message: str


# endregion

# region Azure AI Agent with MCP


class SemanticKernelMCPAgent:
    """Wraps Azure AI Agent with MCP plugins to handle various tasks."""

    def __init__(self):
        self.agent = None
        self.thread = None
        self.client = None
        self.credential = None
        self.plugin = None

    async def initialize(self, mcp_url: str = os.getenv("SPORTS_NEWS_MCP_URL")):
        """Initialize the agent with Azure credentials and MCP plugin."""
        try:
            # Create Azure credential
            self.credential = DefaultAzureCredential()
            
            # Get Azure AI endpoint from environment
            endpoint = os.getenv("AZURE_AI_AGENT_PROJECT_ENDPOINT")
            if not endpoint:
                raise ValueError("AZURE_AI_AGENT_PROJECT_ENDPOINT environment variable is required")
            
            # Create Azure AI client with endpoint and credential
            self.client = AzureAIAgent.create_client(
                endpoint=endpoint,
                credential=self.credential
            )
            
            # Create the MCP plugin
            self.plugin = MCPStreamableHttpPlugin(
                name="SportsNews",
                url=mcp_url,
            )
            
            # Initialize the plugin
            await self.plugin.__aenter__()
            
            # Get model deployment name from environment
            model_deployment_name = os.getenv("AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME")
            if not model_deployment_name:
                raise ValueError("AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME environment variable is required")
            
            # Create agent definition
            agent_definition = await self.client.agents.create_agent(
                model=model_deployment_name,
                name="SportsNewsAssistant",
                instructions="You are a helpful agent that processes sports news stories. \
                    You have available several tools to assist you.  Use the tool that best fits the user request. \
                    Only use the first 10 'Headline' and 'Link' items from the news story to create your response."
            )

            # Create the agent with MCP plugin
            self.agent = AzureAIAgent(
                client=self.client,
                definition=agent_definition,
                plugins=[self.plugin],
            )
            
            logger.info("MCP Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP Agent: {e}")
            await self.cleanup()
            raise

    
    async def stream(
        self,
        user_input: str,
        session_id: str = None,
    ) -> AsyncIterable[dict[str, Any]]:
        """Stream responses from the Azure AI Agent with MCP plugins.

        Args:
            user_input (str): User input message.
            session_id (str): Unique identifier for the session (optional).

        Yields:
            dict: A dictionary containing the content and task completion status.
        """
        if not self.agent:
            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': 'Agent not initialized. Please call initialize() first.',
            }
            return

        try:
            async for response in self.agent.invoke(
                messages=user_input,
                thread=self.thread,
            ):
                self.thread = response.thread
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': str(response),
                }
            
            # Final completion message
            yield {
                'is_task_complete': True,
                'require_user_input': False,
                'content': 'Task completed successfully.',
            }
        except Exception as e:
            # Enhanced error handling for rate limits
            error_message = self._parse_rate_limit_error(e)
            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': error_message,
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
            if any(keyword in error_str for keyword in ["token", "tpm", "tokens per minute"]):
                return (f"⚠️ **Rate Limit Exceeded: Tokens Per Minute (TPM)**\n\n"
                       f"The request exceeded the allowed tokens per minute limit. "
                       f"This typically means too many tokens are being processed in a short time period.\n\n"
                       f"**Suggestions:**\n"
                       f"• Wait a moment before retrying\n"
                       f"• Reduce the length of your input\n"
                       f"• Break large requests into smaller chunks\n\n"
                       f"**Technical details:** {error_type}: {str(error)}")
            
            # Check for requests per minute (RPM) rate limit
            elif any(keyword in error_str for keyword in ["request", "rpm", "requests per minute", "too many requests"]):
                return (f"⚠️ **Rate Limit Exceeded: Requests Per Minute (RPM)**\n\n"
                       f"The request exceeded the allowed requests per minute limit. "
                       f"This typically means too many API calls are being made in a short time period.\n\n"
                       f"**Suggestions:**\n"
                       f"• Wait a moment before retrying\n"
                       f"• Reduce the frequency of requests\n"
                       f"• Implement request batching if possible\n\n"
                       f"**Technical details:** {error_type}: {str(error)}")
            
            # Check for Azure OpenAI specific patterns within rate limit context
            elif "azure" in error_str and ("openai" in error_str or "cognitive" in error_str):
                return (f"⚠️ **Azure OpenAI Quota Exceeded**\n\n"
                       f"Your Azure OpenAI service quota has been exceeded. This could be either "
                       f"tokens per minute (TPM) or requests per minute (RPM).\n\n"
                       f"**Suggestions:**\n"
                       f"• Check your Azure OpenAI resource usage in the Azure portal\n"
                       f"• Wait for the quota to reset\n"
                       f"• Consider upgrading your Azure OpenAI pricing tier\n\n"
                       f"**Technical details:** {error_type}: {str(error)}")
            
            # Check for OpenAI specific patterns within rate limit context
            elif "openai" in error_str:
                return (f"⚠️ **OpenAI Quota Exceeded**\n\n"
                       f"Your OpenAI API quota has been exceeded. This could be either "
                       f"tokens per minute (TPM) or requests per minute (RPM).\n\n"
                       f"**Suggestions:**\n"
                       f"• Check your OpenAI API usage dashboard\n"
                       f"• Wait for the quota to reset\n"
                       f"• Consider upgrading your OpenAI plan\n\n"
                       f"**Technical details:** {error_type}: {str(error)}")
            
            # Generic rate limit error
            else:
                return (f"⚠️ **Rate Limit Exceeded**\n\n"
                       f"A rate limit has been exceeded, but the specific type could not be determined.\n\n"
                       f"**Suggestions:**\n"
                       f"• Wait a moment before retrying\n"
                       f"• Check your API usage and limits\n"
                       f"• Consider reducing request frequency or size\n\n"
                       f"**Technical details:** {error_type}: {str(error)}")
        
        # Check for usage/limit patterns outside of explicit rate limit context
        elif ("usage" in error_str or "limit" in error_str) and ("openai" in error_str or "azure" in error_str):
            if "azure" in error_str:
                return (f"⚠️ **Azure Service Usage Limit Reached**\n\n"
                       f"Your Azure service has reached its usage limit.\n\n"
                       f"**Suggestions:**\n"
                       f"• Check your Azure resource usage in the Azure portal\n"
                       f"• Wait for the limit to reset\n"
                       f"• Consider upgrading your service tier\n\n"
                       f"**Technical details:** {error_type}: {str(error)}")
            else:
                return (f"⚠️ **OpenAI Usage Limit Reached**\n\n"
                       f"Your OpenAI API has reached its usage limit.\n\n"
                       f"**Suggestions:**\n"
                       f"• Check your OpenAI API usage dashboard\n"
                       f"• Wait for the limit to reset\n"
                       f"• Consider upgrading your OpenAI plan\n\n"
                       f"**Technical details:** {error_type}: {str(error)}")
        
        # Default error handling for non-rate-limit errors
        return f'Error processing request: {str(error)}'

    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.thread:
                await self.thread.delete()
                self.thread = None
                logger.info("Thread deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting thread: {e}")
        
        try:
            if self.agent and self.client:
                await self.client.agents.delete_agent(self.agent.id)
                logger.info("Agent deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting agent: {e}")
        
        try:
            if self.plugin:
                await self.plugin.__aexit__(None, None, None)
                self.plugin = None
                logger.info("MCP plugin cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up MCP plugin: {e}")
        
        try:
            if self.client:
                await self.client.close()
                self.client = None
                logger.info("Client closed successfully")
        except Exception as e:
            logger.error(f"Error closing client: {e}")
        
        try:
            if self.credential:
                await self.credential.close()
                self.credential = None
                logger.info("Credential closed successfully")
        except Exception as e:
            logger.error(f"Error closing credential: {e}")
        
        self.agent = None