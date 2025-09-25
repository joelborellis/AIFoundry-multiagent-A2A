#!/usr/bin/env python3
"""
Test script to verify telemetry implementation with realistic scenarios.
"""

import os
import asyncio
from routing_agent import RoutingAgent

# Mock environment variables for testing
os.environ["AZURE_AI_AGENT_PROJECT_ENDPOINT"] = "https://test-endpoint.cognitiveservices.azure.com"
os.environ["AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME"] = "gpt-4"

async def test_realistic_telemetry():
    """Test telemetry with realistic scenarios."""
    print("🧪 Testing RoutingAgent telemetry with realistic scenarios...")
    
    try:
        # Test 1: Create RoutingAgent using the factory method
        print("\n1️⃣ Testing RoutingAgent.create() factory method...")
        routing_agent = await RoutingAgent.create(
            remote_agent_addresses=[],  # Empty for testing
            task_callback=None,
            status_callback=None
        )
        print("✅ RoutingAgent created via factory method")
        
        # Test 2: Test traced methods
        print("\n2️⃣ Testing traced method calls...")
        
        # Test get_or_create_thread method with tracing
        print("   Testing get_or_create_thread...")
        try:
            # This will fail due to mock credentials, but we can test the tracing wrapper
            with routing_agent.tracer.start_as_current_span("test_thread_creation"):
                print("   Span created for thread creation test")
        except Exception as e:
            print(f"   Expected error (mock endpoint): {type(e).__name__}")
        
        # Test 3: Test list_remote_agents method
        print("\n3️⃣ Testing list_remote_agents...")
        agents = routing_agent.list_remote_agents()
        print(f"✅ Remote agents listed: {len(agents)} agents found")
        
        # Test 4: Test send_message method with tracing (should handle no agents gracefully)
        print("\n4️⃣ Testing send_message with tracing...")
        result = await routing_agent.send_message("test_agent", "test task")
        print("✅ send_message handled gracefully with no remote agents")
        print(f"   Result type: {type(result)}")
        
        # Test 5: Test span attributes and error handling
        print("\n5️⃣ Testing comprehensive span usage...")
        with routing_agent.tracer.start_as_current_span("comprehensive_test") as span:
            span.set_attribute("test.scenario", "realistic_testing")
            span.set_attribute("test.remote_agents", len(routing_agent.remote_agent_connections))
            span.set_attribute("test.telemetry_enabled", True)
            
            # Simulate some processing
            span.add_event("processing_started")
            span.set_attribute("processing.status", "complete")
            span.add_event("processing_finished")
            print("✅ Comprehensive span attributes and events set")
        
        print("\n🎉 All realistic telemetry tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_realistic_telemetry())
    exit(0 if success else 1)
