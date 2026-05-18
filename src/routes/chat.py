"""Chat API routes."""

from pydantic import BaseModel
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
import os

# Initialize router
router = APIRouter()

# Initialize streaming chat model
chat_model = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=os.getenv("OPENAI_API_KEY"),
    streaming=True,
)


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    message: str


class ChatResponse(BaseModel):
    """Response body for chat endpoint."""
    message: str
    response: str


async def stream_chat_response(message: str):
    """Stream chat response from OpenAI."""
    try:
        # Use streaming to get chunks of the response
        for chunk in chat_model.stream(message):
            if chunk.content:
                yield chunk.content
    except Exception as e:
        yield f"Error: {str(e)}"


@router.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """Stream chat endpoint that calls OpenAI."""
    return StreamingResponse(
        stream_chat_response(request.message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.post("/chat/non-stream")
async def chat_non_stream(request: ChatRequest) -> ChatResponse:
    """Non-streaming chat endpoint (returns full response at once)."""
    result = chat_model.invoke(request.message)
    return ChatResponse(message=request.message, response=result.content)


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
