import click
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from a2a.server.agent_execution import AgentExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import (
    DefaultRequestHandler,
)
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    GetTaskRequest,
    GetTaskResponse,
    SendMessageRequest,
    SendMessageResponse,
)

from agent_executor import OpenAIWebSearchAgentExecutor


class A2ARequestHandler(DefaultRequestHandler):
    """A2A Request Handler for the A2A Repo Agent."""

    def __init__(
        self, agent_executor: AgentExecutor, task_store: InMemoryTaskStore
    ):
        super().__init__(agent_executor, task_store)

    async def on_get_task(self, request: GetTaskRequest) -> GetTaskResponse:
        return await super().on_get_task(request)

    async def on_message_send(
        self, request: SendMessageRequest
    ) -> SendMessageResponse:
        return await super().on_message_send(request)


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10001)
def main(host: str, port: int):
    """Start the A2A Repo Agent server.

    This function initializes the A2A Repo Agent server with the specified host and port.
    It creates an agent card with the agent's name, description, version, and capabilities.

    Args:
        host (str): The host address to run the server on.
        port (int): The port number to run the server on.
    """
    capabilities = AgentCapabilities(streaming=True)
    skill_sports = AgentSkill(
        id='sports_results_agent',
        name='Sports Results Agent',
        description='Provides sports results (scores, winner, notable stats) across MLB, NBA, NASCAR, golf, college football.',
        tags=['mlb', 'nba', 'nascar', 'golf', 'college football'],
        examples=[
            'Show score for Pirates game last night',
            'What was the final score of Game 7 NBA Finals and who won?',
            'Who won the 2025 U.S. Open (golf) and where was it played?',
        ],
    )

    agent_card = AgentCard(
        name='SportsResultsAgent',
        description='Returns sports results across major leagues.',
        url=f'http://{host}:{port}/',                # JSON-RPC POST target
        version='1.0.0',
        default_input_modes=['text'],                # snake_case fields
        default_output_modes=['text'],
        #preferred_transport='HTTP+JSONRPC',          # align with JSON-RPC
        capabilities=capabilities,
        skills=[skill_sports],
        # supports_authenticated_extended_card=True, # optional
    )

    task_store = InMemoryTaskStore()
    request_handler = A2ARequestHandler(
        agent_executor=OpenAIWebSearchAgentExecutor(),
        task_store=task_store,
    )

    server = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )
    uvicorn.run(server.build(), host=host, port=port)


if __name__ == '__main__':
    main()