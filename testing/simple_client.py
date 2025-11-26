"""
Simple Python client using only standard library for testing the FastAPI routing agent.

This client connects to the /chat/stream endpoint using urllib and displays 
real-time responses from the Azure AI routing agent.
"""

import json
import urllib.request
import urllib.error
from typing import Optional


class SimpleRoutingAgentClient:
    """Simple client using only standard library for the FastAPI routing agent."""
    
    def __init__(self, base_url: str = "http://localhost:8083"):
        self.base_url = base_url.rstrip('/')
        
    def check_health(self) -> bool:
        """Check if the routing agent API is healthy."""
        try:
            response = urllib.request.urlopen(f"{self.base_url}/health", timeout=10)
            if response.status == 200:
                data = json.loads(response.read().decode())
                return data.get("status") == "healthy"
            return False
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def get_agent_info(self) -> Optional[dict]:
        """Get information about the routing agent."""
        try:
            response = urllib.request.urlopen(f"{self.base_url}/", timeout=10)
            if response.status == 200:
                return json.loads(response.read().decode())
            return None
        except Exception as e:
            print(f"Failed to get agent info: {e}")
            return None
    
    def simple_chat(self, message: str, session_id: Optional[str] = None):
        """Send a simple (non-streaming) chat message."""
        print(f"\n🚀 Sending message: '{message}'")
        print("=" * 60)
        
        payload = {"message": message}
        if session_id:
            payload["session_id"] = session_id
        
        try:
            # Prepare the request
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                f"{self.base_url}/chat",
                data=data,
                headers={
                    'Content-Type': 'application/json',
                    'Content-Length': str(len(data))
                },
                method='POST'
            )
            
            # Send request
            response = urllib.request.urlopen(req, timeout=60)
            
            if response.status == 200:
                response_data = json.loads(response.read().decode())
                print(f"🤖 Response: {response_data.get('response', 'No response')}")
                if response_data.get('session_id'):
                    print(f"📝 Session ID: {response_data['session_id']}")
            else:
                print(f"❌ Error: HTTP {response.status}")
                error_text = response.read().decode()
                print(f"Error details: {error_text}")
                
        except urllib.error.HTTPError as e:
            print(f"❌ HTTP Error {e.code}: {e.reason}")
            try:
                error_details = e.read().decode()
                print(f"Error details: {error_details}")
            except:
                pass
        except Exception as e:
            print(f"❌ Error during simple chat: {e}")


def interactive_mode():
    """Run the client in interactive mode."""
    client = SimpleRoutingAgentClient()
    
    print("🤖 Azure AI Routing Agent Client (Standard Library)")
    print("=" * 60)
    
    # Check health
    print("🔍 Checking agent health...")
    is_healthy = client.check_health()
    if not is_healthy:
        print("❌ Agent is not healthy. Please start the FastAPI server first.")
        print("   Make sure the server is running on http://localhost:8083")
        return
    
    print("✅ Agent is healthy!")
    
    # Get agent info
    agent_info = client.get_agent_info()
    if agent_info:
        print(f"\n📊 Agent Info:")
        agent_status = agent_info.get('agent_status', {})
        print(f"   Azure Agent ID: {agent_status.get('azure_agent_id', 'N/A')}")
        print(f"   Available Remote Agents: {agent_status.get('available_remote_agents', 0)}")
        if agent_status.get('remote_agents'):
            print(f"   Remote Agents: {', '.join(agent_status['remote_agents'])}")
    
    print("\n" + "=" * 60)
    print("💬 Interactive Chat Mode")
    print("Commands:")
    print("  - Type your message and press Enter")
    print("  - Type 'quit' or 'exit' to quit")
    print("  - Type 'help' for this help message")
    print("Note: This version uses simple (non-streaming) responses")
    print("=" * 60)
    
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
                print("  - <message>         - Send message to the routing agent")
                print("  - quit/exit         - Exit the application")
                print("  - help              - Show this help message")
                continue
            
            # Send the message
            client.simple_chat(user_input)
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break


def demo_mode():
    """Run a demo with predefined messages."""
    client = SimpleRoutingAgentClient()
    
    print("🎬 Demo Mode - Testing Routing Agent")
    print("=" * 60)
    
    # Check health
    print("🔍 Checking agent health...")
    is_healthy = client.check_health()
    if not is_healthy:
        print("❌ Agent is not healthy. Please start the FastAPI server first.")
        print("   Make sure the server is running on http://localhost:8083")
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
        client.simple_chat(message)
        
        if i < len(demo_messages):
            print(f"\n⏳ Waiting 3 seconds before next demo...")
            import time
            time.sleep(3)
    
    print("\n🎉 Demo completed!")


def main():
    """Main entry point."""
    print("🤖 Azure AI Routing Agent Client (Standard Library)")
    print("Choose mode:")
    print("1. Interactive mode")
    print("2. Demo mode")
    
    try:
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            interactive_mode()
        elif choice == "2":
            demo_mode()
        else:
            print("❌ Invalid choice. Running interactive mode by default.")
            interactive_mode()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
