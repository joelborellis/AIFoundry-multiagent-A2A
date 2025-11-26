"""
Simple test script using requests library for the published Azure AI Foundry agent.

This script uses AzureCliCredential like the azure_ai_basic.py example.
"""

import asyncio
import json
import os
from typing import Optional

try:
    import requests
    from azure.identity.aio import AzureCliCredential
    
    REQUESTS_AVAILABLE = True
except ImportError as e:
    REQUESTS_AVAILABLE = False
    print(f"Required libraries not available: {e}")
    print("Make sure azure-identity is installed")


async def get_azure_access_token() -> Optional[str]:
    """Get an Azure access token using AzureCliCredential."""
    try:
        async with AzureCliCredential() as credential:
            # Get token for Azure ML resource
            token = await credential.get_token("https://ml.azure.com/.default")
            return token.token
    except Exception as e:
        print(f"Failed to get Azure access token: {e}")
        print("Make sure you're logged in with 'az login'")
        return None


async def test_with_requests(api_key: Optional[str] = None):
    """Test the endpoint using the requests library."""
    
    if not REQUESTS_AVAILABLE:
        print("Required libraries not available. Make sure azure-identity is installed")
        return
    
    endpoint = "https://joel-foundry-project-resource.services.ai.azure.com/api/projects/joel-foundry-project/applications/MicrosoftLearnAgent/protocols/openai/responses?api-version=2025-11-15-preview"
    
    # Try to get authentication
    auth_header = None
    
    if api_key:
        auth_header = f"Bearer {api_key}"
        print("Using provided API key for authentication")
    else:
        # Try Azure CLI credential
        print("Getting Azure CLI credentials...")
        access_token = await get_azure_access_token()
        if access_token:
            auth_header = f"Bearer {access_token}"
            print("✅ Using Azure CLI authentication")
        else:
            print("❌ No authentication method available")
            print("You can either:")
            print("1. Run 'az login' and try again")
            print("2. Provide an API key as environment variable AZURE_AI_API_KEY")
            return
    
    # Test question
    question = "What is Azure AI Foundry and how does it help developers?"
    
    # Prepare request using Responses API format (not chat completions)
    data = {
        "input": question
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": auth_header
    }
    
    print(f"\nTesting endpoint: {endpoint}")
    print(f"Question: {question}")
    print("-" * 80)
    
    try:
        response = requests.post(
            endpoint,
            json=data,
            headers=headers,
            timeout=60
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success! Agent responded:")
            print("-" * 40)
            
            # Responses API format uses 'output' instead of 'choices'
            if 'output' in result:
                content = result['output']
                
                # Parse the response to extract clean text
                if isinstance(content, list):
                    # Look for the final message content
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'message' and item.get('role') == 'assistant':
                            message_content = item.get('content', [])
                            if isinstance(message_content, list):
                                for content_item in message_content:
                                    if content_item.get('type') == 'output_text':
                                        content = content_item.get('text', str(content))
                                        break
                
                print("Agent Response:")
                print(content)
                
                # Show usage if available
                if 'usage' in result:
                    usage = result['usage']
                    print(f"\n📊 Token Usage: {usage}")
            elif 'choices' in result and len(result['choices']) > 0:
                # Fallback for standard chat completions format
                content = result['choices'][0]['message']['content']
                print("Agent Response:")
                print(content)
                
                if 'usage' in result:
                    usage = result['usage']
                    print(f"\n📊 Token Usage: {usage}")
            else:
                print("No response content found")
                print("Full response:")
                print(json.dumps(result, indent=2))
                
        elif response.status_code == 401:
            print("❌ Authentication failed")
            print("Make sure you have the correct API key or are logged in with 'az login'")
            
        elif response.status_code == 403:
            print("❌ Permission denied")
            print("Check that your account has access to this resource")
            
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - check your internet connection")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


async def interactive_test():
    """Interactive mode to test multiple questions."""
    
    if not REQUESTS_AVAILABLE:
        print("Required libraries not available. Make sure azure-identity is installed")
        return
    
    # Get API key from environment or Azure CLI
    api_key = os.getenv("AZURE_AI_API_KEY")
    access_token = None
    
    if not api_key:
        print("Getting Azure CLI credentials for interactive session...")
        access_token = await get_azure_access_token()
    
    if not api_key and not access_token:
        print("No authentication available. Please either:")
        print("1. Set AZURE_AI_API_KEY environment variable")
        print("2. Run 'az login'")
        return
    
    auth_header = f"Bearer {api_key}" if api_key else f"Bearer {access_token}"
    
    endpoint = "https://joel-foundry-project-resource.services.ai.azure.com/api/projects/joel-foundry-project/applications/MicrosoftLearnAgent/protocols/openai/responses?api-version=2025-11-15-preview"
    
    print("🤖 Interactive Agent Chat")
    print("Type 'quit' to exit")
    print("-" * 40)
    
    while True:
        question = input("\n💬 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
            
        if not question:
            continue
        
        try:
            data = {
                "input": question
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": auth_header
            }
            
            print("🤔 Thinking...")
            
            response = requests.post(endpoint, json=data, headers=headers, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle Responses API format
                if 'output' in result:
                    content = result['output']
                    
                    # Parse the response to extract clean text
                    if isinstance(content, list):
                        # Look for the final message content
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'message' and item.get('role') == 'assistant':
                                message_content = item.get('content', [])
                                if isinstance(message_content, list):
                                    for content_item in message_content:
                                        if content_item.get('type') == 'output_text':
                                            content = content_item.get('text', str(content))
                                            break
                    
                    print(f"\n🤖 Agent: {content}")
                elif 'choices' in result and len(result['choices']) > 0:
                    # Fallback for standard format
                    content = result['choices'][0]['message']['content']
                    print(f"\n🤖 Agent: {content}")
                else:
                    print("❌ No response from agent")
            else:
                print(f"❌ Error: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")


async def main():
    """Main async function to run the tests."""
    # Check if API key is available in environment
    api_key = os.getenv("AZURE_AI_API_KEY")
    
    print("🚀 Azure AI Foundry Agent Test")
    print("=" * 40)
    
    # Run basic test
    await test_with_requests(api_key)
    
    # Ask if user wants to try interactive mode
    if REQUESTS_AVAILABLE:
        print("\n" + "=" * 60)
        choice = input("\n🎮 Would you like to try interactive mode? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            await interactive_test()


if __name__ == "__main__":
    asyncio.run(main())