#!/usr/bin/env python3
"""
Test script to verify telemetry implementation in RoutingAgent.
"""

import os
import asyncio
from routing_agent import RoutingAgent

# Mock environment variables for testing
os.environ["AZURE_AI_AGENT_PROJECT_ENDPOINT"] = "https://test-endpoint.cognitiveservices.azure.com"
os.environ["AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME"] = "gpt-4"

async def test_telemetry():
    """Test telemetry initialization and basic functionality."""
    print("🧪 Testing RoutingAgent telemetry implementation...")
    
    try:
        # Test 1: Create RoutingAgent instance
        print("\n1️⃣ Testing RoutingAgent creation...")
        routing_agent = RoutingAgent()
        print("✅ RoutingAgent created successfully")
        
        # Test 2: Check if telemetry tracer is initialized
        print("\n2️⃣ Testing telemetry tracer initialization...")
        if hasattr(routing_agent, 'tracer'):
            print("✅ Tracer attribute exists")
            print(f"   Tracer type: {type(routing_agent.tracer)}")
        else:
            print("❌ Tracer attribute missing")
            
        # Test 3: Test span creation (basic functionality)
        print("\n3️⃣ Testing span creation...")
        with routing_agent.tracer.start_as_current_span("test_span") as span:
            span.set_attribute("test.attribute", "test_value")
            span.set_attribute("test.number", 42)
            print("✅ Span created and attributes set successfully")
        
        # Test 4: Test async component initialization (without real remote agents)
        print("\n4️⃣ Testing async component initialization...")
        await routing_agent._async_init_components([])  # Empty list for testing
        print("✅ Async components initialized successfully")
        
        # Test 5: Test get_root_instruction method
        print("\n5️⃣ Testing get_root_instruction method...")
        instructions = routing_agent.get_root_instruction()
        print(f"✅ Instructions generated ({len(instructions)} characters)")
        print(f"   Preview: {instructions[:100]}...")
        
        print("\n🎉 All telemetry tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_telemetry())
    exit(0 if success else 1)
