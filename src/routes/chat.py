"""Chat API routes."""

from pydantic import BaseModel
from fastapi import APIRouter
from ..graph import create_chat_graph

# Initialize router
router = APIRouter()

# Initialize the chat graph
chat_graph = create_chat_graph()


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    message: str


class ChatResponse(BaseModel):
    """Response body for chat endpoint."""
    message: str
    response: str


@router.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Simple chat endpoint that calls OpenAI."""
    result = chat_graph.invoke({"message": request.message, "response": ""})
    return ChatResponse(message=request.message, response=result["response"])


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
