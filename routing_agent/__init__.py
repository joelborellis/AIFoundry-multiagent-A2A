
"""
FastAPI application for multi-agent routing with Azure AI Agents integration.

This application provides REST API endpoints for interacting with a routing agent
that uses Azure AI Agents for core functionality and delegates tasks to remote agents.

Updated to use dependency injection pattern for better testability and maintainability.
"""

import asyncio
import json
import os
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Optional, Annotated
import uvicorn

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dependencies import (
    get_routing_agent_service, 
    get_routing_agent, 
    get_routing_agent_service_instance,
    RoutingAgentService
)
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
# Create FastAPI app
app = FastAPI(
    title="Azure AI Routing Agent API",
    description="REST API for multi-agent routing with Azure AI Agents integration",
    version="1.0.0",  # Keep same version to avoid frontend confusion
    lifespan=lifespan
)

# Endpoint to reset the thread_id (current_thread) in the backend
@app.post("/reset")
async def reset_thread(routing_agent: Annotated[RoutingAgent, Depends(get_routing_agent)]):
    if hasattr(routing_agent, 'current_thread'):
        routing_agent.current_thread = None
    return {"status": "reset", "thread_id": None}

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root(routing_agent: Annotated[RoutingAgent, Depends(get_routing_agent)]):
    """Root endpoint with API information - maintains exact same response format."""
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
        "message": "Azure AI Routing Agent API",
        "version": "1.0.0",
        "status": "running",
        "agent_status": agent_status
    }


@app.get("/health")
async def health_check(routing_agent: Annotated[RoutingAgent, Depends(get_routing_agent)]):
    """Health check endpoint - maintains exact same response format."""
    is_healthy = routing_agent is not None
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "agent_initialized": is_healthy
    }


async def generate_response_stream(
    message: str, 
    session_id: Optional[str] = None, 
    thread_id: Optional[str] = None,
    routing_agent_service: RoutingAgentService = None
):
    """Generate streaming response - maintains exact same SSE format for frontend."""
    if not routing_agent_service:
        yield f"data: {json.dumps({'type': 'error', 'content': 'Routing agent service not available.'})}\n\n"
        return
    
    routing_agent = await routing_agent_service.get_routing_agent()
    if not routing_agent:
        yield f"data: {json.dumps({'type': 'error', 'content': 'Routing agent not initialized. Please restart the application.'})}\n\n"
        return
    
    # Create a unique request ID and status queue for this request
    request_id = str(uuid.uuid4())
    status_queue = routing_agent_service.create_status_queue(request_id)
    
    try:
        # Send initial status - EXACT same format as before
        yield f"data: {json.dumps({'type': 'status', 'content': '🤖 Agent working...'})}\n\n"
        
        # Create a task to process the user message
        async def process_message():
            try:
                response = await routing_agent.process_user_message(message, thread_id)
                return response
            except Exception as e:
                return {"error": str(e)}
        
        # Start processing the message
        process_task = asyncio.create_task(process_message())
        
        # Monitor for status updates while processing - EXACT same logic
        response = None
        while not process_task.done():
            try:
                # Check for status updates with a short timeout
                status_data = status_queue.get_nowait()
                
                if status_data["status_type"] == "agent_start":
                    content = f"🤖 Delegating to <strong>{status_data['agent_name']}</strong> agent..."
                    yield f"data: {json.dumps({'type': 'status', 'content': content})}\n\n"
                elif status_data["status_type"] == "agent_complete":
                    content = f"✅ <strong>{status_data['agent_name']}</strong> agent completed processing"
                    yield f"data: {json.dumps({'type': 'status', 'content': content})}\n\n"
                    
            except Exception:  # Changed from queue.Empty to Exception for broader compatibility
                # No status updates, just wait a bit
                await asyncio.sleep(0.1)
        
        # Get the final result
        response = await process_task
        
        # Send the final response - EXACT same format
        if response and not isinstance(response, dict) or not response.get("error"):
            yield f"data: {json.dumps({'type': 'response', 'content': response})}\n\n"
        else:
            error_msg = response.get("error", "No response received from the agent.") if isinstance(response, dict) else "No response received from the agent."
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
            
    except Exception as e:
        print(f"Error in generate_response_stream (Type: {type(e)}): {e}")
        traceback.print_exc()
        error_message = f"An error occurred: {str(e)}"
        yield f"data: {json.dumps({'type': 'error', 'content': error_message})}\n\n"
    
    finally:
        # Clean up the status queue
        routing_agent_service.remove_status_queue(request_id)
    
    # Send end of stream marker
    yield f"data: {json.dumps({'type': 'end'})}\n\n"


@app.post("/chat/stream")
async def chat_stream(
    request: MessageRequest,
    service: Annotated[RoutingAgentService, Depends(get_routing_agent_service_instance)]
):
    """Stream chat responses from the routing agent - maintains exact same functionality."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    return StreamingResponse(
        generate_response_stream(
            request.message, 
            request.session_id, 
            request.thread_id,
            service
        ),
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
async def chat(
    request: MessageRequest,
    routing_agent: Annotated[RoutingAgent, Depends(get_routing_agent)]
):
    """Non-streaming chat endpoint - maintains exact same functionality."""
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
    """List available remote agents - maintains exact same functionality."""
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
    
    # Run the FastAPI app
    uvicorn.run(
        "__init__:app",
        host="0.0.0.0",
        port=8083,
        reload=True,
        log_level="info"
    )
