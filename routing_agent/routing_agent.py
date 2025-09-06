import json
import os
import time
import uuid

from typing import Any, Dict, List, Optional, Callable

import httpx

from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    Part,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
)
from remote_agent_connection import (
    RemoteAgentConnections,
    TaskUpdateCallback,
)
from azure.ai.agents import AgentsClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder
from dotenv import load_dotenv


load_dotenv()


class AzureAgentContext:
    """Context class."""
    def __init__(self):
        self.state: Dict[str, Any] = {}

class RoutingAgent:
    """The Routing agent.

    This is the agent responsible for choosing which remote sports agents to send
    tasks to and coordinate their work using multiple different Agent types.
    """

    def __init__(
        self,
        task_callback: TaskUpdateCallback | None = None,
        status_callback: Callable[[str, str], None] | None = None,
    ):
        self.task_callback = task_callback
        self.status_callback = status_callback
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ''
        self.context = AzureAgentContext()
        
        # Initialize Azure AI Agents client
        self.agents_client = AgentsClient(
            endpoint=os.environ["AZURE_AI_AGENT_PROJECT_ENDPOINT"],
            credential=DefaultAzureCredential(),
        )
        self.azure_agent = None
        self.current_thread = None

    async def _async_init_components(
        self, remote_agent_addresses: list[str]
    ) -> None:
        """Asynchronous part of initialization."""
        # Use a single httpx.AsyncClient for all card resolutions for efficiency
        async with httpx.AsyncClient(timeout=30) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(
                    client, address
                )  # Constructor is sync
                try:
                    card = (
                        await card_resolver.get_agent_card()
                    )  # get_agent_card is async

                    remote_connection = RemoteAgentConnections(
                        agent_card=card, agent_url=address
                    )
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card
                except httpx.ConnectError as e:
                    print(
                        f'ERROR: Failed to get agent card from {address}: {e}'
                    )
                except Exception as e:  # Catch other potential errors
                    print(
                        f'ERROR: Failed to initialize connection for {address}: {e}'
                    )

        # Populate self.agents using the logic from original __init__ (via list_remote_agents)
        agent_info = []
        for agent_detail_dict in self.list_remote_agents():
            agent_info.append(json.dumps(agent_detail_dict))
        self.agents = '\n'.join(agent_info)

    @classmethod
    async def create(
        cls,
        remote_agent_addresses: list[str],
        task_callback: TaskUpdateCallback | None = None,
        status_callback: Callable[[str, str], None] | None = None,
    ) -> 'RoutingAgent':
        """Create and asynchronously initialize an instance of the RoutingAgent."""
        instance = cls(task_callback, status_callback)
        await instance._async_init_components(remote_agent_addresses)
        return instance

    def create_agent(self):
        """Create an Azure AI Agent instance."""
        instructions = self.get_root_instruction()
        
        try:
            # Create Azure AI Agent with better error handling
            model_name = os.environ.get("AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini")
            print(f"Creating AIFoundry routing agent with model: {model_name}")
            print(f"Instructions length: {len(instructions)} characters")

            # Only include send_message tool if remote agents are available
            tools = []
            if self.remote_agent_connections:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": "send_message",
                        "description": "Sends a task to a remote agent",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "agent_name": {
                                    "type": "string",
                                    "description": "The name of the agent to send the task to"
                                },
                                "task": {
                                    "type": "string",
                                    "description": "The comprehensive conversation context summary and goal to be achieved"
                                }
                            },
                            "required": ["agent_name", "task"]
                        }
                    }
                })
                print(f"Added send_message tool - {len(self.remote_agent_connections)} remote agents available")
            else:
                print("No remote agents available - running without function tools")
            
            self.azure_agent = self.agents_client.create_agent(
                model=model_name,
                name="routing-agent",
                instructions=instructions,
                tools=tools
            )
            print(f"Created Azure AI agent, agent ID: {self.azure_agent.id}")
            
            return self.azure_agent
            
        except Exception as e:
            print(f"Error creating Azure AI agent: {e}")
            print(f"Model name used: {model_name}")
            print(f"Instructions: {instructions[:200]}...")
            raise

    def get_or_create_thread(self, thread_id: Optional[str] = None):
        """Get an existing thread or create a new one if thread_id is not provided."""
        try:
            if thread_id:
                # Try to get the existing thread
                try:
                    thread = self.agents_client.threads.get(thread_id=thread_id)
                    self.current_thread = thread
                    print(f"Using existing thread, thread ID: {thread.id}")
                    return thread
                except Exception as e:
                    print(f"Failed to get existing thread {thread_id}: {e}")
                    # Fall through to create a new thread
            
            # Create a new thread
            thread = self.agents_client.threads.create()
            self.current_thread = thread
            print(f"Created new thread, thread ID: {thread.id}")
            return thread
            
        except Exception as e:
            print(f"Error creating/getting thread: {e}")
            raise

    def get_current_thread_id(self) -> Optional[str]:
        """Get the current thread ID."""
        return self.current_thread.id if self.current_thread else None

    def get_root_instruction(self) -> str:
        """Generate the root instruction for the RoutingAgent."""
        current_agent = self.check_active_agent()
        available_agents = self.list_remote_agents()
        
        if available_agents:
            agents_info = self.agents
            routing_instructions = """
- Delegate user inquiries to appropriate specialized remote agents
- Connect users with sports_news_agent for sports news requests  
- Connect users with sports_results_agent for sports results requests
- Use the send_message function to route requests to the appropriate agent"""
        else:
            agents_info = "No remote agents currently available"
            routing_instructions = """
- No specialized remote agents are currently available
- Provide helpful general responses directly to users
- Inform users that specialized agents are currently unavailable
- Do NOT use the send_message function when no agents are available"""

        return f"""You are an expert Routing Delegator that helps users with sports information requests.

Your role:
{routing_instructions}

Available Agents: {agents_info}
Currently Active Agent: {current_agent['active_agent']}

Always be helpful and provide useful responses to users.

Always respond in html format."""

    def check_active_agent(self):
        """Check the currently active agent."""
        state = self.context.state
        if (
            'session_id' in state
            and 'session_active' in state
            and state['session_active']
            and 'active_agent' in state
        ):
            return {'active_agent': f'{state["active_agent"]}'}
        return {'active_agent': 'None'}

    def initialize_session(self):
        """Initialize a new session."""
        state = self.context.state
        if 'session_active' not in state or not state['session_active']:
            if 'session_id' not in state:
                state['session_id'] = str(uuid.uuid4())
            state['session_active'] = True

    def list_remote_agents(self):
        """List the available remote agents you can use to delegate the task."""
        if not self.cards:
            return []

        remote_agent_info = []
        for card in self.cards.values():
            print(f'Found agent card: {card.model_dump(exclude_none=True)}')
            print('=' * 100)
            remote_agent_info.append(
                {'name': card.name, 'description': card.description}
            )
        return remote_agent_info

    async def send_message(
        self, agent_name: str, task: str
    ):
        """Sends a task to remote sports agent.

        This will send a message to the remote agent named agent_name.

        Args:
            agent_name: The name of the agent to send the task to.
            task: The comprehensive conversation context summary
                and goal to be achieved regarding user inquiry and purchase request.

        Returns:
            A Task object from the remote agent response.
        """
        # Check if any remote agents are available
        if not self.remote_agent_connections:
            return {
                "error": "No remote agents are currently available. The sports news and results agents are not running.",
                "message": "Please ensure the remote agents are started before trying to route requests."
            }
            
        if agent_name not in self.remote_agent_connections:
            available_agents = list(self.remote_agent_connections.keys())
            return {
                "error": f"Agent '{agent_name}' not found. Available agents: {available_agents}",
                "available_agents": available_agents
            }
        
        state = self.context.state
        state['active_agent'] = agent_name
        
        # Notify about agent execution start via callback
        if self.status_callback:
            self.status_callback("agent_start", agent_name)
        
        client = self.remote_agent_connections[agent_name]

        if not client:
            raise ValueError(f'Client not available for {agent_name}')
        
        task_id = state['task_id'] if 'task_id' in state else str(uuid.uuid4())

        if 'context_id' in state:
            context_id = state['context_id']
        else:
            context_id = str(uuid.uuid4())

        message_id = ''
        metadata = {}
        if 'input_message_metadata' in state:
            metadata.update(**state['input_message_metadata'])
            if 'message_id' in state['input_message_metadata']:
                message_id = state['input_message_metadata']['message_id']
        if not message_id:
            message_id = str(uuid.uuid4())

        payload = {
            'message': {
                'role': 'user',
                'parts': [
                    {'type': 'text', 'text': task}
                ],  # Use the 'task' argument here
                'messageId': message_id,
            },
        }

        if task_id:
            payload['message']['taskId'] = task_id

        if context_id:
            payload['message']['contextId'] = context_id

        message_request = SendMessageRequest(
            id=message_id, params=MessageSendParams.model_validate(payload)
        )
        send_response: SendMessageResponse = await client.send_message(
            message_request=message_request
        )
        print('send_response', send_response.model_dump_json(exclude_none=True, indent=2))

        if not isinstance(send_response.root, SendMessageSuccessResponse):
            print('received non-success response. Aborting get task ')
            return

        if not isinstance(send_response.root.result, Task):
            print('received non-task response. Aborting get task ')
            return

        # Notify about agent execution completion via callback
        if self.status_callback:
            self.status_callback("agent_complete", agent_name)

        return send_response.root.result

    async def process_user_message(self, user_message: str, thread_id: Optional[str] = None) -> str:
        """Process a user message through Azure AI Agent and return the response."""
        if not hasattr(self, 'azure_agent') or not self.azure_agent:
            return "Azure AI Agent not initialized. Please ensure the agent is properly created."
        
        try:
            # Initialize session if needed
            self.initialize_session()
            
            # Get or create thread based on provided thread_id
            thread = self.get_or_create_thread(thread_id)
            
            print(f"Processing message: {user_message[:50]}...")
            
            # Create message in the thread
            message = self.agents_client.messages.create(
                thread_id=thread.id, 
                role="user", 
                content=user_message
            )
            print(f"Created message, message ID: {message.id}")

            # Create and run the agent
            print(f"Creating run with agent ID: {self.azure_agent.id}")
            run = self.agents_client.runs.create(
                thread_id=thread.id, 
                agent_id=self.azure_agent.id
            )
            print(f"Created run, run ID: {run.id}")

            # Poll the run until completion
            max_iterations = 60  # 60 seconds timeout
            iteration = 0
            while run.status in ["queued", "in_progress", "requires_action"] and iteration < max_iterations:
                # Handle function calls if needed
                if run.status == "requires_action":
                    await self._handle_required_actions(run)
                
                time.sleep(1)
                iteration += 1
                run = self.agents_client.runs.get(
                    thread_id=thread.id, 
                    run_id=run.id
                )
                print(f"Run status: {run.status} (iteration {iteration})")

            if iteration >= max_iterations:
                return "Request timed out after 60 seconds. Please try again."

            if run.status == "failed":
                error_info = f"Run error: {run.last_error}"
                print(error_info)
                
                # Try to get more detailed error information
                if hasattr(run, 'last_error') and run.last_error:
                    if hasattr(run.last_error, 'code'):
                        error_info += f" (Code: {run.last_error.code})"
                    if hasattr(run.last_error, 'message'):
                        error_info += f" (Message: {run.last_error.message})"
                
                return f"Error processing request: {error_info}"

            # Get the latest messages
            messages = self.agents_client.messages.list(
                thread_id=thread.id, 
                order=ListSortOrder.DESCENDING
            )
            
            # Return the assistant's response
            for msg in messages:
                if msg.role == "assistant" and msg.text_messages:
                    last_text = msg.text_messages[-1]
                    return last_text.text.value
            
            return "No response received from agent."
            
        except Exception as e:
            error_msg = f"Error in process_user_message: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            # Enhanced error handling for rate limits
            enhanced_error_message = self._parse_rate_limit_error(e)
            return enhanced_error_message

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
        return f"An error occurred while processing your message: {str(error)}"

    async def _handle_required_actions(self, run):
        """Handle function calls required by the Azure AI Agent."""
        try:
            if hasattr(run, 'required_action') and run.required_action:
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []
                
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    print(f"Executing function: {function_name} with args: {function_args}")
                    
                    if function_name == "send_message":
                        try:
                            # Call our send_message method
                            result = await self.send_message(
                                agent_name=function_args["agent_name"],
                                task=function_args["task"]
                            )
                            # Convert result to JSON string
                            output = json.dumps(result.model_dump() if hasattr(result, 'model_dump') else str(result))
                        except Exception as e:
                            output = json.dumps({"error": str(e)})
                    else:
                        output = json.dumps({"error": f"Unknown function: {function_name}"})
                    
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": output
                    })
                
                # Submit the tool outputs
                self.agents_client.runs.submit_tool_outputs(
                    thread_id=self.current_thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
                print(f"Submitted {len(tool_outputs)} tool outputs")
                
        except Exception as e:
            print(f"Error handling required actions: {e}")
            import traceback
            traceback.print_exc()

    def cleanup(self):
        """Clean up Azure AI agent resources."""
        try:
            if hasattr(self, 'azure_agent') and self.azure_agent and hasattr(self, 'agents_client') and self.agents_client:
                self.agents_client.delete_agent(self.azure_agent.id)
                print(f"Deleted Azure AI agent: {self.azure_agent.id}")
        except Exception as e:
            print(f"Error cleaning up agent: {e}")
        finally:
            # Close the client to clean up resources
            if hasattr(self, 'agents_client') and self.agents_client:
                try:
                    self.agents_client.close()
                    print("Azure AI client closed")
                except Exception as e:
                    print(f"Error closing client: {e}")
            
            if hasattr(self, 'azure_agent'):
                self.azure_agent = None
            if hasattr(self, 'current_thread'):
                self.current_thread = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()