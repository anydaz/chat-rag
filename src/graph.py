"""LangGraph definition for chat with RAG."""

from typing import TypedDict
from langchain_openai import ChatOpenAI
from .rag import retrieve_context, load_documents_to_chroma
import os


class ChatState(TypedDict):
    """State for chat messages."""
    message: str
    response: str
    context: str


def create_chat_graph():
    """Create a chat graph with RAG retrieval."""
    
    # Initialize OpenAI chat model with streaming
    chat_model = ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
        streaming=True,
    )
    
    # Load documents to Chroma on initialization
    try:
        load_documents_to_chroma()
    except Exception as e:
        print(f"Warning: Could not load documents to Chroma: {e}")
    
    def retrieve_node(state: ChatState) -> ChatState:
        """Retrieve relevant context from documents."""
        context = retrieve_context(state["message"], top_k=3)
        state["context"] = context
        return state
    
    def chat_node(state: ChatState) -> ChatState:
        """Call OpenAI with the user message and context."""
        # Build prompt with context
        if state["context"]:
            prompt = f"""You are a chatbot representing Andy Darmawan. Your role is to answer professional queries about Andy based on the provided information. Be helpful, professional, and accurate.

Relevant Information:
{state["context"]}

Question: {state["message"]}

Answer:"""
        else:
            prompt = state["message"]
        
        result = chat_model.invoke(prompt)
        state["response"] = result.content
        return state
    
    # Simple sequential execution: retrieve context, then chat
    def execute(input_state: ChatState):
        """Execute retrieval then chat."""
        state = retrieve_node(input_state)
        state = chat_node(state)
        return state
    
    return execute

