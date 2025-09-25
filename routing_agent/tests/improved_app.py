"""
FastAPI application using dependency injection pattern.
"""

import json
import os
import traceback
from contextlib import asynccontextmanager
from typing import Optional, Annotated

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dependencies import get_routing_agent_service, get_routing_agent, RoutingAgentService
from routing_agent import RoutingAgent


class MessageRequest(BaseModel):
    """Request model for sending messages to the routing agent."""
    message: str
    session_id: Optional[str] = None
    thread_id: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Check required environment variables on startup
    required_env_vars = [
        "AZURE_AI_AGENT_PROJECT_ENDPOINT",
        "AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    print("✅ Environment variables validated")
    
    try:
        # Initialize the routing agent service
        service = get_routing_agent_service()
        await service.get_routing_agent()  # Initialize on startup
        print("FastAPI application started successfully")
        
        yield
        
    finally:
        # Cleanup on shutdown
        print("Shutting down application...")
        service = get_routing_agent_service()
        await service.cleanup()
        print("FastAPI application has been shut down.")


# Create FastAPI app
app = FastAPI(
    title="Azure AI Routing Agent API (Dependency Injection)",
    description="REST API for multi-agent routing with dependency injection pattern",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root(routing_agent: Annotated[RoutingAgent, Depends(get_routing_agent)]):
    """Root endpoint with API information."""
    if routing_agent and routing_agent.azure_agent:
        agent_status = {
            "azure_agent_id": routing_agent.azure_agent.id,
            "thread_id": routing_agent.get_current_thread_id(),
            "available_remote_agents": len(routing_agent.remote_agent_connections),
            "remote_agents": list(routing_agent.remote_agent_connections.keys())
        }
    else:
        agent_status = {
            "azure_agent_id": None,
            "thread_id": None,
            "available_remote_agents": 0,
            "remote_agents": []
        }
    
    return {
        "message": "Azure AI Routing Agent API with Dependency Injection",
        "version": "2.0.0",
        "status": "running",
        "agent_status": agent_status
    }


@app.get("/health")
async def health_check(routing_agent: Annotated[RoutingAgent, Depends(get_routing_agent)]):
    """Health check endpoint."""
    is_healthy = routing_agent is not None
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "agent_initialized": is_healthy
    }


@app.post("/chat")
async def chat(
    request: MessageRequest,
    routing_agent: Annotated[RoutingAgent, Depends(get_routing_agent)]
):
    """Non-streaming chat endpoint for simple responses."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        response = await routing_agent.process_user_message(request.message, request.thread_id)
        return {
            "response": response,
            "session_id": request.session_id,
            "thread_id": routing_agent.get_current_thread_id()
        }
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/agents")
async def list_agents(routing_agent: Annotated[RoutingAgent, Depends(get_routing_agent)]):
    """List available remote agents."""
    try:
        remote_agents = routing_agent.list_remote_agents()
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
    
    uvicorn.run(
        "improved_app:app",
        host="0.0.0.0",
        port=8084,
        reload=True,
        log_level="info"
    )
