"""LangGraph definition for chat."""

from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI


class ChatState(TypedDict):
    """State for chat messages."""
    message: str
    response: str


def create_chat_graph():
    """Create a simple chat graph with OpenAI."""
    
    # Initialize OpenAI chat model
    chat_model = ChatOpenAI(model="gpt-3.5-turbo")
    
    def chat_node(state: ChatState) -> ChatState:
        """Call OpenAI with the user message."""
        result = chat_model.invoke(state["message"])
        state["response"] = result.content
        return state
    
    # Create the graph
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    graph.set_finish_point("chat")
    
    return graph.compile()
