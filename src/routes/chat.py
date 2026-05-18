"""Chat API routes with RAG."""

from pydantic import BaseModel
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from ..rag import retrieve_context
from ..graph import create_chat_graph
import os

# Initialize router
router = APIRouter()

# Initialize the chat graph (which handles RAG + chat logic)
chat_graph = create_chat_graph()

# Also initialize streaming chat model for direct streaming
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
    context: str


async def stream_chat_response(message: str, context: str):
    """Stream chat response from OpenAI with RAG context."""
    try:
        # Build prompt with context (same as graph)
        if context:
            prompt = f"""You are a chatbot representing Andy Darmawan. Your role is to answer professional queries about Andy based on the provided information. Be helpful, professional, and accurate.

Relevant Information:
{context}

Question: {message}

Answer:"""
        else:
            prompt = message
        
        # Stream response using chat model
        for chunk in chat_model.stream(prompt):
            if chunk.content:
                yield chunk.content
    except Exception as e:
        yield f"Error: {str(e)}"


@router.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """Stream chat endpoint with RAG retrieval."""
    # Retrieve relevant context
    context = retrieve_context(request.message, top_k=3)
    
    async def generate():
        async for chunk in stream_chat_response(request.message, context):
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
    result = chat_graph({
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

