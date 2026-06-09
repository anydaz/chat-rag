"""LangGraph definition for chat with RAG and calendar integration."""

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
import os

# Import nodes
from .nodes import (
    guardrails_node,
    detect_calendar_intent,
    validate_calendar_event,
    retrieve_node,
    calendar_node,
    chat_node,
)


class ChatState(TypedDict):
    """State for chat messages."""
    message: str
    response: str
    context: str
    is_valid: bool
    conversation_history: List[dict]
    calendar_action: Optional[str]
    calendar_result: Optional[str]
    calendar_data: Optional[dict]  # Structured calendar data: {event_title, date_time, timezone, email_address}


def create_chat_graph():
    """Create a chat graph with RAG retrieval, guardrails, and calendar integration."""
    
    # Initialize models
    guardrails_model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )
    
    # Wrap nodes to inject guardrails_model
    def guardrails_wrapper(state: ChatState) -> ChatState:
        return guardrails_node(state, guardrails_model)
    
    def calendar_intent_wrapper(state: ChatState) -> ChatState:
        return detect_calendar_intent(state, guardrails_model)
    
    def should_calendar_or_guardrails(state: ChatState) -> str:
        """Router: Calendar agent or guardrails + retrieval."""
        return "validate_calendar" if state.get("calendar_action") == "CREATE" else "guardrails"
    
    def should_retrieve(state: ChatState) -> str:
        """Router: Determine if question passed guardrails."""
        return "retrieve" if state["is_valid"] else "respond"
    
    # Build the LangGraph
    graph = StateGraph(ChatState)
    
    # Add nodes
    graph.add_node("detect_calendar", calendar_intent_wrapper)
    graph.add_node("validate_calendar", validate_calendar_event)
    graph.add_node("guardrails", guardrails_wrapper)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("calendar", calendar_node)
    graph.add_node("chat", chat_node)
    
    # Set entry point - detect calendar first
    graph.set_entry_point("detect_calendar")
    
    # From detect_calendar, route to validate_calendar (if calendar) or guardrails (if not)
    graph.add_conditional_edges(
        "detect_calendar",
        should_calendar_or_guardrails,
        {
            "validate_calendar": "validate_calendar",
            "guardrails": "guardrails"
        }
    )
    
    # From validate_calendar, route to calendar or chat based on is_valid flag
    graph.add_conditional_edges(
        "validate_calendar",
        lambda state: "calendar" if state.get("is_valid") else "respond",
        {
            "calendar": "calendar",
            "respond": "chat"
        }
    )
    
    # From guardrails, conditionally retrieve or respond
    graph.add_conditional_edges(
        "guardrails",
        should_retrieve,
        {
            "retrieve": "retrieve",
            "respond": "chat"
        }
    )
    
    # From retrieve, go to chat
    graph.add_edge("retrieve", "chat")
    
    # From calendar, go to chat
    graph.add_edge("calendar", "chat")
    
    # Chat is the final node
    graph.set_finish_point("chat")
    
    # Compile with memory checkpointer for conversation history
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
