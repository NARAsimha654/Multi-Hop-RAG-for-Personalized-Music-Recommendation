# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uvicorn # For running the app
import sys
import os

# Import the RAG system class
from rag_system import MusicRAGSystem, RAGConfig

# --- Pydantic Models ---
# Define expected request body structure
class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's query or message.")
    # Optional: Include chat history for conversational context
    # History should be a list of {'role': 'human'/'ai', 'content': 'message'}
    # chat_history: Optional[List[Dict[str, str]]] = None

# Define expected response body structure
class ChatResponse(BaseModel):
    answer: str = Field(..., description="The generated response from the RAG system.")
    # Optional: Return updated chat history if managing state client-side
    # updated_history: Optional[List[Dict[str, str]]] = None

# --- Global Variables / Initialization ---
# Load configuration
try:
    rag_config = RAGConfig()
    # Instantiate the RAG system ONCE when the application starts
    # All models and data will be loaded into memory here
    rag_system = MusicRAGSystem(rag_config)
except Exception as e:
    print(f"FATAL: Failed to initialize RAG system on startup: {e}")
    # Exit if the core system fails to load
    sys.exit("RAG system initialization failed.")

# Create FastAPI app instance
app = FastAPI(
    title="Music RAG API",
    description="API for personalized music recommendations using RAG.",
    version="0.1.0"
)

# --- API Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    """
    Handles a user query, generates a response using the RAG system.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Call the RAG system's generation method
        # Pass history if the request model includes it
        # response_text = rag_system.generate_response(request.query, request.chat_history)
        response_text = rag_system.generate_response(request.query) # Simplified without history passing for now

        if response_text is None:
             raise HTTPException(status_code=500, detail="RAG system failed to generate a response.")

        # Return the response
        # Include updated history if managing state client-side
        return ChatResponse(answer=response_text)

    except Exception as e:
        print(f"Error during /chat endpoint processing: {e}")
        # Depending on the error, you might want specific status codes
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/health")
async def health_check():
    """ Simple health check endpoint """
    # Could add checks here (e.g., Neo4j connection)
    return {"status": "ok"}

# --- Cleanup on Shutdown (Optional) ---
@app.on_event("shutdown")
def shutdown_event():
    print("FastAPI application shutting down...")
    if rag_system:
        rag_system.close_neo4j() # Close Neo4j connection gracefully
    print("Cleanup complete.")

# --- Run with Uvicorn ---
# This block allows running directly with `python main.py`
# For production, use `uvicorn main:app --reload` (or without --reload)
if __name__ == "__main__":
    print("Starting FastAPI server with Uvicorn...")
    # Host '0.0.0.0' makes it accessible on your network, use '127.0.0.1' for local only
    # Reload=True is useful for development, remove for production
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

