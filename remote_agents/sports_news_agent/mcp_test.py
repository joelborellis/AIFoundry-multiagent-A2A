from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
import asyncio
import os
from dotenv import load_dotenv
import tiktoken
from datetime import datetime

load_dotenv()

def count_tokens_for_gpt4_mini(text: str) -> int:
    """
    Count the number of tokens in the text for GPT-4.1-mini model.
    Uses cl100k_base encoding which is used by GPT-4 models.
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error counting tokens: {e}")
        # Fallback: rough estimate of 4 characters per token
        return len(text) // 4

def save_to_file(content: str, filename: str = None) -> str:
    """
    Save content to a .txt file with timestamp.
    Returns the filename used.
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mcp_results_{timestamp}.txt"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Results saved to: {filename}")
        return filename
    except Exception as e:
        print(f"Error saving to file: {e}")
        return None

async def main():
    url=os.getenv("SPORTS_NEWS_MCP_URL")

    # Connect to a streamable HTTP server this is to connect to the mcp server

    async with streamablehttp_client(url=url) as (
        read_stream,
        write_stream,
        _,
    ):
        # Create a session using the client streams
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()
            
            print("getting session")
            # List available tools
            tools_result = await session.list_tools()
            print("Available tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")

            # Call our calculator tool
            result = await session.call_tool("get_nascar_news")
            
            # Extract the text content from the result
            result_text = result.content[0].text
            print(f"{result_text}")
            
            # Save results to file
            filename = save_to_file(result_text)
            
            # Count tokens for GPT-4.1-mini
            token_count = count_tokens_for_gpt4_mini(result_text)
            
            # Print summary information
            print(f"\n--- Summary ---")
            print(f"Content length: {len(result_text)} characters")
            print(f"Estimated tokens for GPT-4.1-mini: {token_count}")
            if filename:
                print(f"Results saved to: {filename}")
            
            # Also save summary information to file
            if filename:
                summary_filename = filename.replace(".txt", "_summary.txt")
                summary_content = f"""MCP Tool Call Results Summary
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Tool called: get_nascar_news
Content length: {len(result_text)} characters
Estimated tokens for GPT-4.1-mini: {token_count}

Original content saved to: {filename}
"""
                save_to_file(summary_content, summary_filename)
            
if __name__ == "__main__":
    asyncio.run(main())