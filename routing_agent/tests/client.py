"""
Simple Python client for testing the FastAPI routing agent streaming endpoint.

This client connects to the /chat/stream endpoint and displays real-time responses
from the Azure AI routing agent.
"""

import asyncio
import json
import sys
from typing import Optional

import httpx


class RoutingAgentClient:
    """Client for interacting with the FastAPI routing agent."""
    
    def __init__(self, base_url: str = "http://localhost:8083"):
        self.base_url = base_url.rstrip('/')
        
    async def check_health(self) -> bool:
        """Check if the routing agent API is healthy."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    data = response.json()
                    return data.get("status") == "healthy"
                return False
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    async def get_agent_info(self) -> Optional[dict]:
        """Get information about the routing agent."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/")
                if response.status_code == 200:
                    return response.json()
                return None
        except Exception as e:
            print(f"Failed to get agent info: {e}")
            return None
    
    async def stream_chat(self, message: str, session_id: Optional[str] = None):
        """Stream chat responses from the routing agent."""
        print(f"\n🚀 Sending message: '{message}'")
        print("=" * 60)
        
        payload = {"message": message}
        if session_id:
            payload["session_id"] = session_id
        
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/stream",
                    json=payload,
                    headers={"Accept": "text/event-stream"}
                ) as response:
                    
                    if response.status_code != 200:
                        print(f"❌ Error: HTTP {response.status_code}")
                        error_text = await response.aread()
                        print(f"Error details: {error_text.decode()}")
                        return
                    
                    print("📡 Streaming response:")
                    print("-" * 40)
                    
                    async for chunk in response.aiter_text():
                        # Handle Server-Sent Events format
                        lines = chunk.strip().split('\n')
                        for line in lines:
                            if line.startswith('data: '):
                                try:
                                    data_str = line[6:]  # Remove 'data: ' prefix
                                    if data_str.strip():
                                        data = json.loads(data_str)
                                        await self._handle_stream_data(data)
                                except json.JSONDecodeError:
                                    print(f"⚠️  Invalid JSON: {line}")
                                except Exception as e:
                                    print(f"⚠️  Error processing line: {e}")
                    
                    print("\n" + "=" * 60)
                    print("✅ Stream completed")
                    
        except httpx.TimeoutException:
            print("❌ Request timed out")
        except Exception as e:
            print(f"❌ Error during streaming: {e}")
    
    async def _handle_stream_data(self, data: dict):
        """Handle individual stream data chunks."""
        message_type = data.get('type', 'unknown')
        content = data.get('content', '')
        
        if message_type == 'status':
            print(f"🔄 Status: {content}")
        elif message_type == 'response':
            print(f"🤖 Response: {content}")
        elif message_type == 'error':
            print(f"❌ Error: {content}")
        elif message_type == 'end':
            print("🏁 End of stream")
        else:
            print(f"📦 {message_type}: {content}")
    
    async def simple_chat(self, message: str, session_id: Optional[str] = None):
        """Send a simple (non-streaming) chat message."""
        print(f"\n🚀 Sending simple message: '{message}'")
        print("=" * 60)
        
        payload = {"message": message}
        if session_id:
            payload["session_id"] = session_id
        
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{self.base_url}/chat",
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"🤖 Response: {data.get('response', 'No response')}")
                    if data.get('session_id'):
                        print(f"📝 Session ID: {data['session_id']}")
                else:
                    print(f"❌ Error: HTTP {response.status_code}")
                    print(f"Error details: {response.text}")
                    
        except Exception as e:
            print(f"❌ Error during simple chat: {e}")


async def interactive_mode():
    """Run the client in interactive mode."""
    client = RoutingAgentClient()
    
    print("🤖 Azure AI Routing Agent Client")
    print("=" * 50)
    
    # Check health
    print("🔍 Checking agent health...")
    is_healthy = await client.check_health()
    if not is_healthy:
        print("❌ Agent is not healthy. Please start the FastAPI server first.")
        return
    
    print("✅ Agent is healthy!")
    
    # Get agent info
    agent_info = await client.get_agent_info()
    if agent_info:
        print(f"\n📊 Agent Info:")
        print(f"   Status: {agent_info.get('status', 'unknown')}")
        agent_status = agent_info.get('agent_status', {})
        print(f"   Azure Agent ID: {agent_status.get('azure_agent_id', 'N/A')}")
        print(f"   Available Remote Agents: {agent_status.get('available_remote_agents', 0)}")
        if agent_status.get('remote_agents'):
            print(f"   Remote Agents: {', '.join(agent_status['remote_agents'])}")
    
    print("\n" + "=" * 50)
    print("💬 Interactive Chat Mode")
    print("Commands:")
    print("  - Type your message and press Enter")
    print("  - Type 'stream:' followed by your message for streaming")
    print("  - Type 'simple:' followed by your message for simple response")
    print("  - Type 'quit' or 'exit' to quit")
    print("  - Type 'help' for this help message")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n🎤 You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
                
            if user_input.lower() == 'help':
                print("\n📖 Commands:")
                print("  - stream:<message>  - Send message with streaming response")
                print("  - simple:<message>  - Send message with simple response")
                print("  - <message>         - Default to streaming response")
                print("  - quit/exit         - Exit the application")
                continue
            
            # Parse command
            if user_input.startswith('stream:'):
                message = user_input[7:].strip()
                if message:
                    await client.stream_chat(message)
                else:
                    print("⚠️  Please provide a message after 'stream:'")
                    
            elif user_input.startswith('simple:'):
                message = user_input[7:].strip()
                if message:
                    await client.simple_chat(message)
                else:
                    print("⚠️  Please provide a message after 'simple:'")
                    
            else:
                # Default to streaming
                await client.stream_chat(user_input)
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break


async def demo_mode():
    """Run a demo with predefined messages."""
    client = RoutingAgentClient()
    
    print("🎬 Demo Mode - Testing Routing Agent")
    print("=" * 50)
    
    # Check health
    print("🔍 Checking agent health...")
    is_healthy = await client.check_health()
    if not is_healthy:
        print("❌ Agent is not healthy. Please start the FastAPI server first.")
        return
    
    print("✅ Agent is healthy!")
    
    # Demo messages
    demo_messages = [
        "Hello! Can you help me?",
        "What sports agents are available?",
        "Get me the latest sports news",
        "What are the recent sports results?",
    ]
    
    for i, message in enumerate(demo_messages, 1):
        print(f"\n🎯 Demo {i}/{len(demo_messages)}")
        await client.stream_chat(message)
        
        if i < len(demo_messages):
            print("\n⏳ Waiting 3 seconds before next demo...")
            await asyncio.sleep(3)
    
    print("\n🎉 Demo completed!")


async def main():
    """Main entry point."""
    print("🤖 Azure AI Routing Agent Client")
    print("Choose mode:")
    print("1. Interactive mode")
    print("2. Demo mode")
    
    try:
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            await interactive_mode()
        elif choice == "2":
            await demo_mode()
        else:
            print("❌ Invalid choice. Running interactive mode by default.")
            await interactive_mode()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
