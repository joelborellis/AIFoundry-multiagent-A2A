from collections.abc import Callable

import httpx
import json

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
    """A class to hold the connections to the remote agents."""

    def __init__(self, agent_card: AgentCard, agent_url: str):
        #print(f'agent_card: {agent_card}')
        #print(f'agent_url: {agent_url}')
        self._httpx_client = httpx.AsyncClient(timeout=30)
<<<<<<< HEAD
        
        # Construct the correct endpoint URL based on the agent card
        # If the agent card URL already includes /rpc/v1, use it; otherwise add it
        card_url = agent_card.url.rstrip('/')
        if card_url.endswith('/rpc/v1'):
            endpoint_url = f'{card_url}/message:send'
        else:
            # Try the non-streaming endpoint first
            endpoint_url = f'{card_url}/rpc/v1/message:send'
        
        print(f'Using endpoint URL: {endpoint_url}')
        self.agent_client = A2AClient(
            self._httpx_client, agent_card, url=endpoint_url
=======
        # A2A server uses "/" as the default RPC endpoint, not "/message:send"
        self.agent_client = A2AClient(
            self._httpx_client, agent_card, url=f'{agent_url}/'
>>>>>>> 141e55db27ca12e2027cc2d3d295890c7897c316
        )
        
        # Store fallback URL in case the primary endpoint fails
        self.fallback_url = f'{card_url}/'  # Root endpoint for streaming
        self.card = agent_card

    def get_agent(self) -> AgentCard:
        return self.card

    async def send_message(
        self, message_request: SendMessageRequest
    ) -> SendMessageResponse:
        try:
            # Try the primary endpoint first
            return await self.agent_client.send_message(message_request)
        except Exception as primary_error:
            print(f"Primary endpoint failed: {primary_error}")
            
            # Try fallback to root endpoint if available
            if hasattr(self, 'fallback_url'):
                try:
                    print(f"Trying fallback endpoint: {self.fallback_url}")
                    
                    # Create a new A2A client for the fallback URL
                    fallback_client = A2AClient(
                        self._httpx_client, self.card, url=self.fallback_url
                    )
                    return await fallback_client.send_message(message_request)
                    
                except Exception as fallback_error:
                    print(f"Fallback endpoint also failed: {fallback_error}")
                    
                    # Try direct HTTP call as final fallback
                    try:
                        return await self._direct_http_fallback(message_request)
                    except Exception as http_error:
                        print(f"Direct HTTP fallback also failed: {http_error}")
                        # Re-raise the original error if all fallbacks fail
                        raise primary_error
            else:
                # No fallback available, re-raise the original error
                raise primary_error
    
    async def _direct_http_fallback(self, message_request: SendMessageRequest) -> SendMessageResponse:
        """Direct HTTP fallback when A2A client fails."""
        base_url = self.card.url.rstrip('/')
        
        # Try the non-streaming endpoint first
        endpoints_to_try = [
            f"{base_url}/rpc/v1/message:send",
            f"{base_url}/",  # Root streaming endpoint
        ]
        
        for endpoint in endpoints_to_try:
            try:
                print(f"Trying direct HTTP to: {endpoint}")
                
                # Convert the message request to JSON-RPC format
                payload = {
                    "jsonrpc": "2.0",
                    "id": message_request.id,
                    "method": "message.send",
                    "params": message_request.params.model_dump()
                }
                
                response = await self._httpx_client.post(
                    endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                response.raise_for_status()
                
                # Parse the JSON-RPC response
                response_data = response.json()
                
                # Convert back to SendMessageResponse format
                from a2a.types import SendMessageResponse
                return SendMessageResponse.model_validate(response_data)
                
            except Exception as e:
                print(f"Direct HTTP to {endpoint} failed: {e}")
                continue
        
        raise Exception("All direct HTTP endpoints failed")