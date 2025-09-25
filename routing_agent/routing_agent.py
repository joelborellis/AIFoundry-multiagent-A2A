import json
import os
import time
from pydantic import BaseModel
from typing import List, Literal, Optional
import uuid

from typing import Any, Dict, Optional, Callable

import httpx

from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
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
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace
from dotenv import load_dotenv

# Enable Azure tracing with content recording
os.environ["AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED"] = "true"

load_dotenv()


class Part(BaseModel):
    kind: Literal["text"]
    text: str


class Artifact(BaseModel):
    artifactId: str
    description: Optional[str] = None
    name: Optional[str] = None
    parts: List[Part] = []


class Message(BaseModel):
    contextId: str
    kind: Literal["message"]
    messageId: str
    parts: List[Part]
    role: str
    taskId: str


class Status(BaseModel):
    state: str


class Result(BaseModel):
    artifacts: List[Artifact] = []
    contextId: str
    history: List[Message] = []
    id: str
    kind: str
    status: Status


class Envelope(BaseModel):
    id: str
    jsonrpc: str
    result: Result


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
        self.agents: str = ""
        self.context = AzureAgentContext()

        # Rate limiting tracking
        self.request_count = 0
        self.rate_limit_errors = 0
        self.last_request_time = None

        # Initialize telemetry tracing
        self._initialize_telemetry()

        # Initialize Azure AI Agents client
        self.agents_client = AgentsClient(
            endpoint=os.environ["AZURE_AI_AGENT_PROJECT_ENDPOINT"],
            credential=DefaultAzureCredential(),
        )
        self.azure_agent = None
        self.current_thread = None

    def _initialize_telemetry(self):
        """Initialize Azure Monitor telemetry for tracing."""
        try:
            # Initialize AI Project client for telemetry
            project_client = AIProjectClient(
                credential=DefaultAzureCredential(),
                endpoint=os.environ["AZURE_AI_AGENT_PROJECT_ENDPOINT"],
            )

            # Get Application Insights connection string and configure monitoring
            connection_string = (
                project_client.telemetry.get_application_insights_connection_string()
            )
            configure_azure_monitor(connection_string=connection_string)

            # Initialize tracer
            self.tracer = trace.get_tracer(__name__)
            print("✅ Telemetry tracing initialized successfully")

        except Exception as e:
            print(f"⚠️ Warning: Failed to initialize telemetry tracing: {e}")
            # Create a no-op tracer if telemetry fails
            self.tracer = trace.get_tracer(__name__)

    async def _async_init_components(self, remote_agent_addresses: list[str]) -> None:
        """Asynchronous part of initialization."""
        with self.tracer.start_as_current_span("init_remote_agent_connections") as span:
            span.set_attribute("remote_agents.count", len(remote_agent_addresses))
            span.set_attribute("remote_agents.addresses", str(remote_agent_addresses))

            # Use a single httpx.AsyncClient for all card resolutions for efficiency
            async with httpx.AsyncClient(timeout=30) as client:
                successful_connections = 0
                failed_connections = 0

                for address in remote_agent_addresses:
                    with self.tracer.start_as_current_span(
                        "connect_remote_agent"
                    ) as agent_span:
                        agent_span.set_attribute("agent.address", address)

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

                            agent_span.set_attribute("agent.name", card.name)
                            agent_span.set_attribute("agent.success", True)
                            successful_connections += 1

                        except httpx.ConnectError as e:
                            agent_span.set_attribute("agent.success", False)
                            agent_span.set_attribute("error.type", "connection_error")
                            agent_span.set_attribute("error.message", str(e))
                            failed_connections += 1
                            print(
                                f"ERROR: Failed to get agent card from {address}: {e}"
                            )
                        except Exception as e:  # Catch other potential errors
                            agent_span.set_attribute("agent.success", False)
                            agent_span.set_attribute("error.type", "general_error")
                            agent_span.set_attribute("error.message", str(e))
                            failed_connections += 1
                            print(
                                f"ERROR: Failed to initialize connection for {address}: {e}"
                            )

            span.set_attribute("connections.successful", successful_connections)
            span.set_attribute("connections.failed", failed_connections)

            # Populate self.agents using the logic from original __init__ (via list_remote_agents)
            agent_info = []
            for agent_detail_dict in self.list_remote_agents():
                agent_info.append(json.dumps(agent_detail_dict))
            self.agents = "\n".join(agent_info)

            span.set_attribute("agents.info_generated", True)
            span.set_attribute("agents.total_available", len(self.cards))

    @classmethod
    async def create(
        cls,
        remote_agent_addresses: list[str],
        task_callback: TaskUpdateCallback | None = None,
        status_callback: Callable[[str, str], None] | None = None,
    ) -> "RoutingAgent":
        """Create and asynchronously initialize an instance of the RoutingAgent."""
        instance = cls(task_callback, status_callback)
        await instance._async_init_components(remote_agent_addresses)
        return instance

    def create_agent(self):
        """Create an Azure AI Agent instance."""
        with self.tracer.start_as_current_span("create_azure_agent") as span:
            instructions = self.get_root_instruction()

            try:
                # Create Azure AI Agent with better error handling
                model_name = os.environ.get(
                    "AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini"
                )

                # Add span attributes for better observability
                span.set_attribute("agent.model", model_name)
                span.set_attribute("agent.instructions_length", len(instructions))
                span.set_attribute(
                    "agent.remote_agents_count", len(self.remote_agent_connections)
                )

                print(f"Creating AIFoundry routing agent with model: {model_name}")
                print(f"Instructions length: {len(instructions)} characters")

                # Only include send_message tool if remote agents are available
                tools = []
                if self.remote_agent_connections:
                    tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": "send_message",
                                "description": "Sends a task to a remote agent",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "agent_name": {
                                            "type": "string",
                                            "description": "The name of the agent to send the task to",
                                        },
                                        "task": {
                                            "type": "string",
                                            "description": "The comprehensive conversation context summary and goal to be achieved",
                                        },
                                    },
                                    "required": ["agent_name", "task"],
                                },
                            },
                        }
                    )
                    span.set_attribute("agent.tools_enabled", True)
                    print(
                        f"Added send_message tool - {len(self.remote_agent_connections)} remote agents available"
                    )
                else:
                    span.set_attribute("agent.tools_enabled", False)
                    print("No remote agents available - running without function tools")

                self.azure_agent = self.agents_client.create_agent(
                    model=model_name,
                    name="routing-agent",
                    instructions=instructions,
                    tools=tools,
                )

                span.set_attribute("agent.id", self.azure_agent.id)
                span.set_attribute("agent.created", True)
                print(f"Created Azure AI agent, agent ID: {self.azure_agent.id}")

                return self.azure_agent

            except Exception as e:
                span.set_attribute("agent.created", False)
                span.set_attribute("error.message", str(e))
                span.record_exception(e)
                print(f"Error creating Azure AI agent: {e}")
                print(f"Model name used: {model_name}")
                print(f"Instructions: {instructions[:200]}...")
                raise

    def get_or_create_thread(self, thread_id: Optional[str] = None):
        """Get an existing thread or create a new one if thread_id is not provided."""
        with self.tracer.start_as_current_span("get_or_create_thread") as span:
            try:
                span.set_attribute("thread.requested_id", thread_id or "new")

                if thread_id:
                    # Try to get the existing thread
                    try:
                        thread = self.agents_client.threads.get(thread_id=thread_id)
                        self.current_thread = thread
                        span.set_attribute("thread.action", "retrieved_existing")
                        span.set_attribute("thread.id", thread.id)
                        print(f"Using existing thread, thread ID: {thread.id}")
                        return thread
                    except Exception as e:
                        span.set_attribute("thread.retrieval_failed", True)
                        span.set_attribute("thread.retrieval_error", str(e))
                        print(f"Failed to get existing thread {thread_id}: {e}")
                        # Fall through to create a new thread

                # Create a new thread
                thread = self.agents_client.threads.create()
                self.current_thread = thread
                span.set_attribute("thread.action", "created_new")
                span.set_attribute("thread.id", thread.id)
                print(f"Created new thread, thread ID: {thread.id}")
                return thread

            except Exception as e:
                span.set_attribute("thread.error", str(e))
                span.record_exception(e)
                print(f"Error creating/getting thread: {e}")
                raise

    def get_current_thread_id(self) -> Optional[str]:
        """Get the current thread ID."""
        return self.current_thread.id if self.current_thread else None

    def get_root_instruction(self) -> str:
        """Generate the root instruction for the RoutingAgent."""
        current_agent = self.check_active_agent()
        # Use the already computed self.cards instead of calling list_remote_agents again
        has_available_agents = bool(self.cards)

        if has_available_agents:
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
            "session_id" in state
            and "session_active" in state
            and state["session_active"]
            and "active_agent" in state
        ):
            return {"active_agent": f'{state["active_agent"]}'}
        return {"active_agent": "None"}

    def initialize_session(self):
        """Initialize a new session."""
        state = self.context.state
        if "session_active" not in state or not state["session_active"]:
            if "session_id" not in state:
                state["session_id"] = str(uuid.uuid4())
            state["session_active"] = True

    def list_remote_agents(self):
        """List the available remote agents you can use to delegate the task."""
        if not self.cards:
            return []

        remote_agent_info = []
        for card in self.cards.values():
            print(f"Found agent card: {card.model_dump(exclude_none=True)}")
            print("=" * 100)
            remote_agent_info.append(
                {"name": card.name, "description": card.description}
            )
        return remote_agent_info

    async def send_message(self, agent_name: str, task: str):
        """Sends a task to remote sports agent.

        This will send a message to the remote agent named agent_name.

        Args:
            agent_name: The name of the agent to send the task to.
            task: The comprehensive conversation context summary
                and goal to be achieved regarding user inquiry and purchase request.

        Returns:
            A Task object from the remote agent response.
        """
        with self.tracer.start_as_current_span("send_message_to_remote_agent") as span:
            span.set_attribute("remote_agent.name", agent_name)
            span.set_attribute("task.length", len(task))
            span.set_attribute("task.word_count", len(task.split()))

            # Check if any remote agents are available
            if not self.remote_agent_connections:
                span.set_attribute("error.type", "no_remote_agents")
                return {
                    "error": "No remote agents are currently available. The sports news and results agents are not running.",
                    "message": "Please ensure the remote agents are started before trying to route requests.",
                }

            if agent_name not in self.remote_agent_connections:
                available_agents = list(self.remote_agent_connections.keys())
                span.set_attribute("error.type", "agent_not_found")
                span.set_attribute("available_agents", str(available_agents))
                return {
                    "error": f"Agent '{agent_name}' not found. Available agents: {available_agents}",
                    "available_agents": available_agents,
                }

            state = self.context.state
            state["active_agent"] = agent_name

            # Notify about agent execution start via callback
            if self.status_callback:
                self.status_callback("agent_start", agent_name)

            client = self.remote_agent_connections[agent_name]

            if not client:
                span.set_attribute("error.type", "client_unavailable")
                raise ValueError(f"Client not available for {agent_name}")

            task_id = state["task_id"] if "task_id" in state else str(uuid.uuid4())

            if "context_id" in state:
                context_id = state["context_id"]
            else:
                context_id = str(uuid.uuid4())

            message_id = ""
            metadata = {}
            if "input_message_metadata" in state:
                metadata.update(**state["input_message_metadata"])
                if "message_id" in state["input_message_metadata"]:
                    message_id = state["input_message_metadata"]["message_id"]
            if not message_id:
                message_id = str(uuid.uuid4())

            span.set_attribute("message.id", message_id)
            span.set_attribute("task.id", task_id)
            span.set_attribute("context.id", context_id)

            payload = {
                "message": {
                    "role": "user",
                    "parts": [
                        {"type": "text", "text": task}
                    ],  # Use the 'task' argument here
                    "messageId": message_id,
                },
            }

            if task_id:
                payload["message"]["taskId"] = task_id

            if context_id:
                payload["message"]["contextId"] = context_id

            with self.tracer.start_as_current_span("remote_agent_call") as call_span:
                call_span.set_attribute("remote_agent.name", agent_name)
                call_span.set_attribute("payload.message_id", message_id)

                message_request = SendMessageRequest(
                    id=message_id, params=MessageSendParams.model_validate(payload)
                )
                send_response: SendMessageResponse = await client.send_message(
                    message_request=message_request
                )
                print(
                    "send_response",
                    send_response.model_dump_json(exclude_none=True, indent=2),
                )

                if not isinstance(send_response.root, SendMessageSuccessResponse):
                    call_span.set_attribute("success", False)
                    call_span.set_attribute("error.type", "non_success_response")
                    print("received non-success response. Aborting get task ")
                    return

                if not isinstance(send_response.root.result, Task):
                    call_span.set_attribute("success", False)
                    call_span.set_attribute("error.type", "non_task_response")
                    print("received non-task response. Aborting get task ")
                    return

                # History as (role, text) pairs
                agent_response = send_response.model_dump_json(
                    exclude_none=True, indent=2
                )
                env = Envelope.model_validate_json(agent_response)

                # Now you have typed dot-access:
                task_id = env.result.id
                state = env.result.status.state
                context_id = env.result.contextId
                artifact_text = (
                    env.result.artifacts[0].parts[0].text
                )  # "Task completed successfully."
                first_user_msg = next(
                    (m for m in env.result.history if m.role == "user"), None
                )
                first_user_txt = (
                    " ".join(p.text for p in first_user_msg.parts)
                    if first_user_msg
                    else None
                )

                # Last agent message text (often the “answer”)
                last_agent_msg = next(
                    (m for m in reversed(env.result.history) if m.role == "agent"), None
                )
                last_agent_txt = (
                    " ".join(p.text for p in last_agent_msg.parts)
                    if last_agent_msg
                    else None
                )

                captured = {
                    "envelope_id": env.id,
                    "task_id": task_id,
                    "state": state,
                    "context_id": context_id,
                    "artifact_text": artifact_text,
                    "first_user_text": first_user_txt,
                    "last_agent_text": last_agent_txt,
                }

                print(f"captured: {captured}")

                call_span.set_attribute("agent_response", captured)
                call_span.set_attribute("success", True)
                span.set_attribute("success", True)

            # Notify about agent execution completion via callback
            if self.status_callback:
                self.status_callback("agent_complete", agent_name)

            return send_response.root.result

    async def process_user_message(
        self, user_message: str, thread_id: Optional[str] = None
    ) -> str:
        """Process a user message through Azure AI Agent and return the response."""
        with self.tracer.start_as_current_span("process_user_message") as span:
            if not hasattr(self, "azure_agent") or not self.azure_agent:
                span.set_attribute("error.type", "agent_not_initialized")
                return "Azure AI Agent not initialized. Please ensure the agent is properly created."

            try:
                # Add span attributes for input tracking
                span.set_attribute("message.length", len(user_message))
                span.set_attribute("message.word_count", len(user_message.split()))
                span.set_attribute(
                    "message.estimated_tokens", len(user_message.split()) * 1.3
                )
                span.set_attribute("thread.requested_id", thread_id or "new")

                # Initialize session if needed
                self.initialize_session()

                # Get or create thread based on provided thread_id
                thread = self.get_or_create_thread(thread_id)
                span.set_attribute("thread.id", thread.id)

                print(f"Processing message: {user_message[:50]}...")
                print(f"Message length: {len(user_message)} characters")
                print(
                    f"Estimated tokens: ~{len(user_message.split()) * 1.3:.0f} (rough estimate)"
                )

                # Create message in the thread
                with self.tracer.start_as_current_span(
                    "create_message"
                ) as message_span:
                    # ...existing code...
                    message = self.agents_client.messages.create(
                        thread_id=thread.id, role="user", content=user_message
                    )
                    message_span.set_attribute("message.id", message.id)
                    span.set_attribute("message.id", message.id)
                    print(f"Created message, message ID: {message.id}")

                # Create and run the agent
                with self.tracer.start_as_current_span(
                    "create_and_run_agent"
                ) as run_span:
                    print(f"Creating run with agent ID: {self.azure_agent.id}")
                    print(
                        f"Model: {os.environ.get('AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME', 'unknown')}"
                    )

                    # Add timestamp for rate limit tracking
                    import datetime

                    start_time = datetime.datetime.now()
                    run_span.set_attribute("run.start_time", start_time.isoformat())
                    print(f"Run started at: {start_time}")

                    # ...existing code...
                    run = self.agents_client.runs.create(
                        thread_id=thread.id, agent_id=self.azure_agent.id
                    )
                    run_span.set_attribute("run.id", run.id)
                    span.set_attribute("run.id", run.id)
                    print(f"Created run, run ID: {run.id}")

                    # Poll the run until completion with adaptive polling
                    with self.tracer.start_as_current_span(
                        "poll_run_completion"
                    ) as poll_span:
                        max_iterations = 60  # 60 seconds timeout
                        iteration = 0

                        while (
                            run.status in ["queued", "in_progress", "requires_action"]
                            and iteration < max_iterations
                        ):
                            # Handle function calls if needed
                            if run.status == "requires_action":
                                with self.tracer.start_as_current_span(
                                    "handle_required_actions"
                                ):
                                    await self._handle_required_actions(run)

                            # Use fixed sleep time now that rate tracking is removed
                            sleep_time = 2.0
                            time.sleep(sleep_time)
                            iteration += 1

                            try:
                                run = self.agents_client.runs.get(
                                    thread_id=thread.id, run_id=run.id
                                )
                                print(
                                    f"Run status is: {run.status} (iteration {iteration}, slept {sleep_time:.1f}s)"
                                )
                            except Exception as e:
                                print(
                                    f"Error getting run status (iteration {iteration}): {e}"
                                )
                                # Check if this is a rate limit error
                                if (
                                    "rate" in str(e).lower()
                                    or "limit" in str(e).lower()
                                ):
                                    pass
                                # If we can't get status, wait longer and try again
                                time.sleep(5)
                                continue

                        poll_span.set_attribute("poll.iterations", iteration)
                        poll_span.set_attribute("poll.final_status", run.status)

                        if iteration >= max_iterations:
                            span.set_attribute("error.type", "timeout")
                            span.set_attribute("run.timeout", True)
                            return (
                                "Request timed out after 60 seconds. Please try again."
                            )

                        if run.status == "failed":
                            # Track rate limit errors
                            error_str = str(run.last_error) if run.last_error else ""
                            if any(
                                term in error_str.lower()
                                for term in ["rate", "limit", "quota", "throttle"]
                            ):
                                # ...existing code...
                                span.set_attribute("error.type", "rate_limit")
                            else:
                                span.set_attribute("error.type", "run_failed")

                            span.set_attribute("run.failed", True)
                            span.set_attribute("run.error", error_str)

                            error_info = f"Run error: {run.last_error}"
                            print(error_info)

                            # Enhanced debugging for rate limit errors
                            if hasattr(run, "last_error") and run.last_error:
                                print(f"Full error object: {run.last_error}")
                                print(f"Error type: {type(run.last_error)}")

                                error_details = {}
                                if hasattr(run.last_error, "code"):
                                    error_details["code"] = run.last_error.code
                                    error_info += f" (Code: {run.last_error.code})"
                                if hasattr(run.last_error, "message"):
                                    error_details["message"] = run.last_error.message
                                    error_info += (
                                        f" (Message: {run.last_error.message})"
                                    )
                                if hasattr(run.last_error, "type"):
                                    error_details["type"] = run.last_error.type
                                    error_info += f" (Type: {run.last_error.type})"
                                if hasattr(run.last_error, "param"):
                                    error_details["param"] = run.last_error.param
                                    error_info += f" (Param: {run.last_error.param})"

                                print(f"Error details extracted: {error_details}")

                                # Check if this is a rate limit error and provide specific guidance
                                if (
                                    error_details.get("code") == "rate_limit_exceeded"
                                    or "rate limit" in str(run.last_error).lower()
                                    or "quota" in str(run.last_error).lower()
                                ):

                                    # Provide specific rate limit troubleshooting
                                    rate_limit_msg = self._analyze_rate_limit_error(
                                        run.last_error, error_details
                                    )
                                    return rate_limit_msg

                            return f"Error processing request: {error_info}"

                # Get the latest messages and return response
                with self.tracer.start_as_current_span(
                    "get_response_messages"
                ) as response_span:
                    # ...existing code...
                    messages = self.agents_client.messages.list(
                        thread_id=thread.id, order=ListSortOrder.DESCENDING
                    )

                    # Return the assistant's response
                    for msg in messages:
                        if msg.role == "assistant" and msg.text_messages:
                            last_text = msg.text_messages[-1]
                            response_content = last_text.text.value
                            response_span.set_attribute(
                                "response.length", len(response_content)
                            )
                            response_span.set_attribute(
                                "response.word_count", len(response_content.split())
                            )
                            span.set_attribute("success", True)
                            return response_content

                    span.set_attribute("success", False)
                    span.set_attribute("error.type", "no_response")
                    return "No response received from agent."

            except Exception as e:
                span.set_attribute("success", False)
                span.set_attribute("error.message", str(e))
                span.record_exception(e)
                error_msg = f"Error in process_user_message: {e}"
                print(error_msg)
                import traceback

                traceback.print_exc()

    async def _handle_required_actions(self, run):
        """Handle function calls required by the Azure AI Agent."""
        try:
            if hasattr(run, "required_action") and run.required_action:
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    print(
                        f"Executing function: {function_name} with args: {function_args}"
                    )

                    if function_name == "send_message":
                        try:
                            # Call our send_message method
                            result = await self.send_message(
                                agent_name=function_args["agent_name"],
                                task=function_args["task"],
                            )
                            # Convert result to JSON string
                            output = json.dumps(
                                result.model_dump()
                                if hasattr(result, "model_dump")
                                else str(result)
                            )
                        except Exception as e:
                            output = json.dumps({"error": str(e)})
                    else:
                        output = json.dumps(
                            {"error": f"Unknown function: {function_name}"}
                        )

                    tool_outputs.append(
                        {"tool_call_id": tool_call.id, "output": output}
                    )

                # Submit the tool outputs
                self.agents_client.runs.submit_tool_outputs(
                    thread_id=self.current_thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs,
                )
                print(f"Submitted {len(tool_outputs)} tool outputs")

        except Exception as e:
            print(f"Error handling required actions: {e}")
            import traceback

            traceback.print_exc()

    def cleanup(self):
        """Clean up Azure AI agent resources."""
        try:
            if (
                hasattr(self, "azure_agent")
                and self.azure_agent
                and hasattr(self, "agents_client")
                and self.agents_client
            ):
                self.agents_client.delete_agent(self.azure_agent.id)
                print(f"Deleted Azure AI agent: {self.azure_agent.id}")
        except Exception as e:
            print(f"Error cleaning up agent: {e}")
        finally:
            # Close the client to clean up resources
            if hasattr(self, "agents_client") and self.agents_client:
                try:
                    self.agents_client.close()
                    print("Azure AI client closed")
                except Exception as e:
                    print(f"Error closing client: {e}")

            if hasattr(self, "azure_agent"):
                self.azure_agent = None
            if hasattr(self, "current_thread"):
                self.current_thread = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
