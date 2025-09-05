"""
FastAPI application for multi-agent routing with Azure AI Agents integration.

This application provides REST API endpoints for interacting with a routing agent
that uses Azure AI Agents for core functionality and delegates tasks to remote agents.
"""

import asyncio
import json
import os
import traceback
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from routing_agent import RoutingAgent


class MessageRequest(BaseModel):
    """Request model for sending messages to the routing agent."""
    message: str
    session_id: Optional[str] = None


class RoutingAgentManager:
    """Manager class for the routing agent lifecycle."""
    
    def __init__(self):
        self.routing_agent: Optional[RoutingAgent] = None
    
    async def initialize_routing_agent(self):
        """Initialize the Azure AI routing agent."""
        try:
            print("Initializing Azure AI routing agent looking for remote Agent cards...")
            
            # Create the routing agent with remote agent addresses
            self.routing_agent = await RoutingAgent.create(
                remote_agent_addresses=[
                    os.getenv('SPORTS_RESULTS_URL', 'http://localhost:10001'),
                    #os.getenv('SPORTS_NEWS_URL', 'http://localhost:10002'),
                ]
            )
            
            # Create the Azure AI agent
            azure_agent = self.routing_agent.create_agent()
            print(f"Azure AI routing agent initialized successfully with Name and ID: {azure_agent.name} - {azure_agent.id}")

        except Exception as e:
            print(f"Failed to initialize routing agent: {e}")
            traceback.print_exc()
            raise
    
    async def cleanup_routing_agent(self):
        """Clean up the routing agent resources."""
        if self.routing_agent:
            try:
                self.routing_agent.cleanup()
                print("Routing agent cleaned up successfully.")
            except Exception as e:
                print(f"Error during cleanup: {e}")
            finally:
                self.routing_agent = None


# Global manager instance
agent_manager = RoutingAgentManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Check required environment variables on startup
    required_env_vars = [
        "AZURE_AI_AGENT_PROJECT_ENDPOINT",
        "AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME"
    ]
    
    # Check if application credentials are provided
    app_creds = ["AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET", "AZURE_TENANT_ID"]
    has_app_creds = all(os.getenv(var) for var in app_creds)
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    if has_app_creds:
        print("✅ Using Azure application (service principal) authentication")
    else:
        print("✅ Using DefaultAzureCredential authentication")
        print("   Make sure you're logged in with 'az login' or have other valid credentials")
    
    try:
        # Initialize the routing agent
        await agent_manager.initialize_routing_agent()
        print("FastAPI application started successfully")
        
        yield
        
    finally:
        # Cleanup on shutdown
        print("Shutting down application...")
        await agent_manager.cleanup_routing_agent()
        print("FastAPI application has been shut down.")


# Create FastAPI app
app = FastAPI(
    title="Azure AI Routing Agent API",
    description="REST API for multi-agent routing with Azure AI Agents integration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    if agent_manager.routing_agent and agent_manager.routing_agent.azure_agent:
        agent_status = {
            "azure_agent_id": agent_manager.routing_agent.azure_agent.id,
            "thread_id": agent_manager.routing_agent.current_thread.id if agent_manager.routing_agent.current_thread else None,
            "available_remote_agents": len(agent_manager.routing_agent.remote_agent_connections),
            "remote_agents": list(agent_manager.routing_agent.remote_agent_connections.keys())
        }
    else:
        agent_status = {
            "azure_agent_id": None,
            "thread_id": None,
            "available_remote_agents": 0,
            "remote_agents": []
        }
    
    return {
        "message": "Azure AI Routing Agent API",
        "version": "1.0.0",
        "status": "running",
        "agent_status": agent_status
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    is_healthy = agent_manager.routing_agent is not None
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "agent_initialized": is_healthy
    }


async def generate_response_stream(message: str, session_id: Optional[str] = None):
    """Generate streaming response from the routing agent."""
    if not agent_manager.routing_agent:
        yield f"data: {json.dumps({'error': 'Routing agent not initialized. Please restart the application.'})}\n\n"
        return
    
    try:
        # Process the message through Azure AI Agent
        response = await agent_manager.routing_agent.process_user_message(message)
        
        # Send the final response
        if response:
            yield f"data: {json.dumps({'type': 'response', 'content': response})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'error', 'content': 'No response received from the agent.'})}\n\n"
            
    except Exception as e:
        print(f"Error in generate_response_stream (Type: {type(e)}): {e}")
        traceback.print_exc()
        error_message = f"An error occurred: {str(e)}"
        yield f"data: {json.dumps({'type': 'error', 'content': error_message})}\n\n"
    
    # Send end of stream marker
    yield f"data: {json.dumps({'type': 'end'})}\n\n"


@app.post("/chat/stream")
async def chat_stream(request: MessageRequest):
    """Stream chat responses from the routing agent."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    return StreamingResponse(
        generate_response_stream(request.message, request.session_id),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@app.post("/chat")
async def chat(request: MessageRequest):
    """Non-streaming chat endpoint for simple responses."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if not agent_manager.routing_agent:
        raise HTTPException(
            status_code=503, 
            detail="Routing agent not initialized. Please restart the application."
        )
    
    try:
        response = await agent_manager.routing_agent.process_user_message(request.message)
        return {
            "response": response,
            "session_id": request.session_id
        }
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/agents")
async def list_agents():
    """List available remote agents."""
    if not agent_manager.routing_agent:
        raise HTTPException(
            status_code=503,
            detail="Routing agent not initialized"
        )
    
    try:
        remote_agents = agent_manager.routing_agent.list_remote_agents()
        return {
            "remote_agents": remote_agents,
            "count": len(remote_agents)
        }
    except Exception as e:
        print(f"Error listing agents: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI app
    uvicorn.run(
        "__init__:app",
        host="0.0.0.0",
        port=8083,
        reload=True,
        log_level="info"
    )
