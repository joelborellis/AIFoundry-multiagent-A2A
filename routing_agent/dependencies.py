"""
Enhanced dependency injection with full status queue support.
Maintains complete compatibility with existing frontend.
"""

import os
import queue
import threading
from functools import lru_cache
from typing import Annotated, Dict, Callable, Optional, TYPE_CHECKING

from fastapi import Depends

if TYPE_CHECKING:
    from routing_agent import RoutingAgent
else:
    from routing_agent import RoutingAgent


class StatusQueueManager:
    """Manages status queues for real-time updates."""
    
    def __init__(self):
        self.status_queues: Dict[str, queue.Queue] = {}
        self._queue_lock = threading.Lock()
    
    def create_status_queue(self, request_id: str) -> queue.Queue:
        """Create a status queue for a specific request."""
        with self._queue_lock:
            status_queue = queue.Queue()
            self.status_queues[request_id] = status_queue
            return status_queue
    
    def remove_status_queue(self, request_id: str):
        """Remove a status queue for a specific request."""
        with self._queue_lock:
            self.status_queues.pop(request_id, None)
    
    def broadcast_status(self, status_type: str, agent_name: str):
        """Broadcast status to all active queues."""
        status_data = {
            "type": "agent_status",
            "status_type": status_type,
            "agent_name": agent_name
        }
        
        with self._queue_lock:
            for request_id, status_queue in list(self.status_queues.items()):
                try:
                    status_queue.put_nowait(status_data)
                except queue.Full:
                    # If queue is full, remove it (client likely disconnected)
                    self.status_queues.pop(request_id, None)


class RoutingAgentService:
    """Enhanced service class with status queue management."""
    
    def __init__(self):
        self._routing_agent: Optional["RoutingAgent"] = None
        self.status_queue_manager = StatusQueueManager()
    
    def status_callback(self, status_type: str, agent_name: str):
        """Callback to handle status updates from the routing agent."""
        self.status_queue_manager.broadcast_status(status_type, agent_name)
    
    async def get_routing_agent(self) -> "RoutingAgent":
        """Get or create the routing agent instance."""
        if self._routing_agent is None:
            from routing_agent import RoutingAgent
            self._routing_agent = await RoutingAgent.create(
                remote_agent_addresses=[
                    os.getenv('SPORTS_RESULTS_URL', 'http://localhost:10001'),
                    #os.getenv('SPORTS_NEWS_URL', 'http://localhost:10002'),
                ],
                status_callback=self.status_callback  # Pass the callback
            )
            # Create the Azure AI agent
            self._routing_agent.create_agent()
        
        return self._routing_agent
    
    def create_status_queue(self, request_id: str) -> queue.Queue:
        """Create a status queue for a request."""
        return self.status_queue_manager.create_status_queue(request_id)
    
    def remove_status_queue(self, request_id: str):
        """Remove a status queue for a request."""
        self.status_queue_manager.remove_status_queue(request_id)
    
    async def cleanup(self):
        """Clean up the routing agent resources."""
        if self._routing_agent:
            self._routing_agent.cleanup()
            self._routing_agent = None


@lru_cache()
def get_routing_agent_service() -> RoutingAgentService:
    """Get the singleton routing agent service."""
    return RoutingAgentService()


async def get_routing_agent(
    service: Annotated[RoutingAgentService, Depends(get_routing_agent_service)]
) -> "RoutingAgent":
    """Dependency to get the routing agent instance."""
    return await service.get_routing_agent()


async def get_routing_agent_service_instance(
    service: Annotated[RoutingAgentService, Depends(get_routing_agent_service)]
) -> RoutingAgentService:
    """Dependency to get the service instance directly."""
    return service
