"""
Direct test of the routing agent connection to sports agent.
This bypasses the Azure AI agents authentication and tests just the connection fix.
"""

import asyncio
import httpx
import sys
import os

# Add the parent directory to Python path so we can import routing_agent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a2a.client import A2ACardResolver
from routing_agent.remote_agent_connection import RemoteAgentConnections
from routing_agent.routing_agent import RoutingAgent


async def test_direct_connection():
    """Test the direct connection to sports agent."""
    print("🧪 Testing direct connection to sports agent...")
    
    # Test connection to sports agent
    sports_agent_url = "http://localhost:10001"
    
    try:
        print(f"📡 Connecting to sports agent at {sports_agent_url}")
        
        async with httpx.AsyncClient(timeout=30) as client:
            # Get the agent card first
            card_resolver = A2ACardResolver(client, sports_agent_url)
            card = await card_resolver.get_agent_card()
            print(f"✅ Successfully got agent card: {card.name}")
            
            # Create connection
            remote_connection = RemoteAgentConnections(
                agent_card=card, 
                agent_url=sports_agent_url
            )
            print(f"✅ Created remote connection")
            
            # Create a minimal routing agent context
            class MockContext:
                def __init__(self):
                    self.state = {}
            
            # Test message sending
            print(f"📤 Sending test message...")
            
            # Create minimal routing agent instance for testing
            routing_agent = RoutingAgent()
            routing_agent.context = MockContext()
            routing_agent.remote_agent_connections = {"SportsResultsAgent": remote_connection}
            routing_agent.cards = {"SportsResultsAgent": card}
            
            # Initialize basic tracer for testing
            from opentelemetry import trace
            routing_agent.tracer = trace.get_tracer(__name__)
            
            # Test the send_message method
            result = await routing_agent.send_message(
                "SportsResultsAgent", 
                "Who won the Steelers game last night?"
            )
            
            print(f"✅ Successfully got response!")
            print(f"📋 Result type: {type(result)}")
            
            # Check if we got a valid result
            if result:
                print(f"🎉 Test successful! Connection is working.")
                
                # Check if task_id and context_id are now stored in state
                state = routing_agent.context.state
                print(f"📝 State after call: {state}")
                
                if "task_id" in state and "context_id" in state:
                    print(f"✅ task_id and context_id properly stored: task_id={state['task_id']}, context_id={state['context_id']}")
                else:
                    print(f"❌ task_id or context_id not stored properly")
            else:
                print(f"❌ No result received")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_direct_connection())