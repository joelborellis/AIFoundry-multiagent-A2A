"""
Simple startup script for the FastAPI routing agent application.
"""

import uvicorn

if __name__ == "__main__":
    # Run the FastAPI app
    uvicorn.run(
        "__init__:app",
        host="0.0.0.0",
        port=8083,
        reload=True,
        log_level="info"
    )
