"""
Test-friendly implementation examples.
"""

import pytest
from unittest.mock import AsyncMock, Mock
from fastapi.testclient import TestClient

from dependencies import RoutingAgentService, get_routing_agent_service
from improved_app import app
from routing_agent import RoutingAgent


class MockRoutingAgentService(RoutingAgentService):
    """Mock routing agent service for testing."""
    
    def __init__(self):
        super().__init__()
        self._mock_agent = AsyncMock(spec=RoutingAgent)
    
    async def get_routing_agent(self) -> RoutingAgent:
        """Return a mock routing agent."""
        return self._mock_agent


def test_dependency_injection():
    """Test dependency injection with mocked service."""
    
    # Override the dependency
    def get_mock_service():
        return MockRoutingAgentService()
    
    app.dependency_overrides[get_routing_agent_service] = get_mock_service
    
    try:
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        
    finally:
        # Clean up override
        app.dependency_overrides.clear()


@pytest.fixture
async def routing_agent_service():
    """Pytest fixture for routing agent service."""
    service = RoutingAgentService()
    yield service
    await service.cleanup()


@pytest.fixture
def mock_routing_agent():
    """Pytest fixture for mock routing agent."""
    mock_agent = AsyncMock(spec=RoutingAgent)
    mock_agent.azure_agent = Mock()
    mock_agent.azure_agent.id = "test-agent-id"
    mock_agent.get_current_thread_id.return_value = "test-thread-id"
    mock_agent.remote_agent_connections = {"test_agent": Mock()}
    mock_agent.process_user_message.return_value = "Test response"
    return mock_agent


class TestRoutingAgentAPI:
    """Test class demonstrating testable patterns."""
    
    def test_with_mock_agent(self, mock_routing_agent):
        """Test endpoint with mocked routing agent."""
        
        def get_mock_agent():
            return mock_routing_agent
        
        app.dependency_overrides[get_routing_agent_service] = lambda: MockRoutingAgentService()
        
        try:
            client = TestClient(app)
            response = client.get("/")
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Azure AI Routing Agent API with Dependency Injection"
            
        finally:
            app.dependency_overrides.clear()
    
    def test_chat_endpoint(self, mock_routing_agent):
        """Test chat endpoint with mocked agent."""
        
        def get_mock_service():
            service = MockRoutingAgentService()
            service._mock_agent = mock_routing_agent
            return service
        
        app.dependency_overrides[get_routing_agent_service] = get_mock_service
        
        try:
            client = TestClient(app)
            response = client.post("/chat", json={"message": "Hello"})
            
            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            
            # Verify the mock was called
            mock_routing_agent.process_user_message.assert_called_once()
            
        finally:
            app.dependency_overrides.clear()


# Example of integration test
@pytest.mark.asyncio
async def test_routing_agent_service_integration():
    """Integration test for the routing agent service."""
    service = RoutingAgentService()
    
    try:
        # This would require proper environment setup
        # agent = await service.get_routing_agent()
        # assert agent is not None
        pass
    finally:
        await service.cleanup()
