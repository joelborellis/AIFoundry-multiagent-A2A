from collections.abc import Callable
import httpx

from a2a.client import A2AClient
from a2a.types import (
    AgentCard,
    SendMessageRequest,
    SendMessageResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)
from dotenv import load_dotenv

load_dotenv()

TaskCallbackArg = Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
TaskUpdateCallback = Callable[[TaskCallbackArg, AgentCard], Task]


class RemoteAgentConnections:
    """Thin wrapper around A2AClient that uses the Agent Card's base URL."""

    def __init__(self, agent_card: AgentCard, agent_url: str | None = None):
        # Honor the card’s advertised URL; ensure trailing slash for JSON-RPC POST target.
        base_url = agent_card.url.rstrip("/") + "/"

        # Single shared async client; bump timeout if your agent can run searches.
        self._httpx_client = httpx.AsyncClient(timeout=60)

        # Let the SDK handle routing/method naming (message/send) at the base URL.
        self.agent_client = A2AClient(self._httpx_client, agent_card, url=base_url)
        self.card = agent_card

    def get_agent(self) -> AgentCard:
        return self.card

    async def send_message(self, message_request: SendMessageRequest) -> SendMessageResponse:
        return await self.agent_client.send_message(message_request)

    async def aclose(self) -> None:
        """Close the underlying HTTP client when you're done."""
        await self._httpx_client.aclose()