"""
Factory pattern with configuration for the routing agent.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol

from routing_agent import RoutingAgent


@dataclass
class RoutingAgentConfig:
    """Configuration for the routing agent."""
    sports_results_url: str = "http://localhost:10001"
    sports_news_url: str = "http://localhost:10002"
    azure_endpoint: Optional[str] = None
    model_deployment_name: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "RoutingAgentConfig":
        """Create configuration from environment variables."""
        return cls(
            sports_results_url=os.getenv('SPORTS_RESULTS_URL', cls.sports_results_url),
            sports_news_url=os.getenv('SPORTS_NEWS_URL', cls.sports_news_url),
            azure_endpoint=os.getenv('AZURE_AI_AGENT_PROJECT_ENDPOINT'),
            model_deployment_name=os.getenv('AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME')
        )
    
    def validate(self) -> None:
        """Validate configuration."""
        if not self.azure_endpoint:
            raise ValueError("AZURE_AI_AGENT_PROJECT_ENDPOINT is required")
        if not self.model_deployment_name:
            raise ValueError("AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME is required")


class RoutingAgentFactory(ABC):
    """Abstract factory for creating routing agents."""
    
    @abstractmethod
    async def create_routing_agent(self, config: RoutingAgentConfig) -> RoutingAgent:
        """Create a routing agent instance."""
        pass


class DefaultRoutingAgentFactory(RoutingAgentFactory):
    """Default implementation of the routing agent factory."""
    
    async def create_routing_agent(self, config: RoutingAgentConfig) -> RoutingAgent:
        """Create a routing agent instance."""
        config.validate()
        
        routing_agent = await RoutingAgent.create(
            remote_agent_addresses=[
                config.sports_results_url,
                config.sports_news_url,
            ]
        )
        
        # Create the Azure AI agent
        routing_agent.create_agent()
        
        return routing_agent


class RoutingAgentManager:
    """Manager for routing agent lifecycle using factory pattern."""
    
    def __init__(self, factory: RoutingAgentFactory, config: RoutingAgentConfig):
        self.factory = factory
        self.config = config
        self._routing_agent: Optional[RoutingAgent] = None
    
    async def get_routing_agent(self) -> RoutingAgent:
        """Get or create the routing agent."""
        if self._routing_agent is None:
            self._routing_agent = await self.factory.create_routing_agent(self.config)
        return self._routing_agent
    
    async def cleanup(self) -> None:
        """Clean up the routing agent."""
        if self._routing_agent:
            self._routing_agent.cleanup()
            self._routing_agent = None
    
    async def reset(self) -> None:
        """Reset the routing agent (useful for configuration changes)."""
        await self.cleanup()
        # Next call to get_routing_agent() will create a new instance


# Example usage with dependency injection
def create_routing_agent_manager() -> RoutingAgentManager:
    """Factory function to create the routing agent manager."""
    config = RoutingAgentConfig.from_env()
    factory = DefaultRoutingAgentFactory()
    return RoutingAgentManager(factory, config)
