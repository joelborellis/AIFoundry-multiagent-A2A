import logging
import uvicorn

import click
import httpx

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from agent_executor import OpenAIWebSearchAgentExecutor
from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


@click.command()
@click.option('--host', default='localhost')
@click.option('--port', default=10001)
def main(host, port):
    """Starts the OpenAI Agents server using A2A."""
    request_handler = DefaultRequestHandler(
        agent_executor=OpenAIWebSearchAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=get_agent_card(host, port), http_handler=request_handler
    )

    uvicorn.run(server.build(), host=host, port=port)


# ...imports unchanged...

def get_agent_card(host: str, port: int):
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
        preferred_transport='HTTP+JSONRPC',          # align with JSON-RPC
        capabilities=capabilities,
        skills=[skill_sports],
        # supports_authenticated_extended_card=True, # optional
    )
    return agent_card


if __name__ == '__main__':
    main()