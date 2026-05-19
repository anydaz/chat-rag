"""Chat API routes with RAG."""

from pydantic import BaseModel
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from ..graph import create_chat_graph
import asyncio

# Initialize router
router = APIRouter()

# Initialize the chat graph (which handles RAG + guardrails + chat logic)
chat_graph = create_chat_graph()


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    message: str


class ChatResponse(BaseModel):
    """Response body for chat endpoint."""
    message: str
    response: str
    context: str


async def stream_chat_response(message: str):
    """Stream chat response from graph with guardrails + RAG."""
    try:
        # Use graph to get response (handles guardrails + retrieval + chat)
        result = chat_graph.invoke({
            "message": message,
            "response": "",
            "context": ""
        })
        
        # Stream response character by character
        response_text = result.get("response", "")
        for char in response_text:
            yield char
            await asyncio.sleep(0)  # Yield control to allow streaming
    except Exception as e:
        yield f"Error: {str(e)}"


@router.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """Stream chat endpoint with guardrails + RAG retrieval."""
    async def generate():
        async for chunk in stream_chat_response(request.message):
            yield chunk
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.post("/chat/non-stream")
async def chat_non_stream(request: ChatRequest) -> ChatResponse:
    """Non-streaming chat endpoint with RAG retrieval."""
    # Use the graph to get response with RAG
    result = chat_graph.invoke({
        "message": request.message,
        "response": "",
        "context": ""
    })
    
    return ChatResponse(
        message=request.message,
        response=result["response"],
        context=result["context"]
    )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

