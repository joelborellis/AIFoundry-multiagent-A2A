"""
Application state pattern for the routing agent FastAPI application.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional, Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from routing_agent import RoutingAgent


class AppState:
    """Application state container."""
    
    def __init__(self):
        self.routing_agent: Optional[RoutingAgent] = None
    
    async def initialize(self):
        """Initialize the routing agent."""
        self.routing_agent = await RoutingAgent.create(
            remote_agent_addresses=[
                os.getenv('SPORTS_RESULTS_URL', 'http://localhost:10001'),
                os.getenv('SPORTS_NEWS_URL', 'http://localhost:10002'),
            ]
        )
        self.routing_agent.create_agent()
    
    async def cleanup(self):
        """Clean up resources."""
        if self.routing_agent:
            self.routing_agent.cleanup()
            self.routing_agent = None


class MessageRequest(BaseModel):
    """Request model for sending messages to the routing agent."""
    message: str
    session_id: Optional[str] = None
    thread_id: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    app.state.app_state = AppState()
    
    try:
        await app.state.app_state.initialize()
        print("Application state initialized successfully")
        yield
    finally:
        await app.state.app_state.cleanup()
        print("Application state cleaned up")


# Create FastAPI app
app = FastAPI(
    title="Azure AI Routing Agent API (App State)",
    description="REST API using FastAPI application state pattern",
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_routing_agent(request: Request) -> RoutingAgent:
    """Get the routing agent from application state."""
    routing_agent = request.app.state.app_state.routing_agent
    if not routing_agent:
        raise HTTPException(
            status_code=503,
            detail="Routing agent not initialized"
        )
    return routing_agent


@app.get("/")
async def root(request: Request):
    """Root endpoint with API information."""
    routing_agent = get_routing_agent(request)
    
    agent_status = {
        "azure_agent_id": routing_agent.azure_agent.id if routing_agent.azure_agent else None,
        "thread_id": routing_agent.get_current_thread_id(),
        "available_remote_agents": len(routing_agent.remote_agent_connections),
        "remote_agents": list(routing_agent.remote_agent_connections.keys())
    }
    
    return {
        "message": "Azure AI Routing Agent API with App State",
        "version": "2.1.0",
        "status": "running",
        "agent_status": agent_status
    }


@app.post("/chat")
async def chat(request_data: MessageRequest, request: Request):
    """Chat endpoint."""
    if not request_data.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    routing_agent = get_routing_agent(request)
    
    try:
        response = await routing_agent.process_user_message(
            request_data.message, 
            request_data.thread_id
        )
        return {
            "response": response,
            "session_id": request_data.session_id,
            "thread_id": routing_agent.get_current_thread_id()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app_state_pattern:app",
        host="0.0.0.0",
        port=8085,
        reload=True,
        log_level="info"
    )
