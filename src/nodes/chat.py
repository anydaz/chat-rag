"""Chat response generation node."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..graph import ChatState

from langchain_openai import ChatOpenAI
import os


def chat_node(state: "ChatState") -> "ChatState":
    """Call OpenAI with the user message and context."""
    # Initialize chat model
    chat_model = ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
        streaming=True,
    )
    
    # Initialize conversation history if not present
    if "conversation_history" not in state:
        state["conversation_history"] = []
    
    # If calendar action was taken, pass through the agent's response directly
    if state.get("calendar_action") and state.get("calendar_result"):
        result = state["calendar_result"]
        # Don't re-process - use the calendar agent's response directly
        # It already handles all cases: asking for missing info, confirming events, etc.
        state["response"] = result
        state["conversation_history"].append({"role": "user", "content": state["message"]})
        state["conversation_history"].append({"role": "assistant", "content": state["response"]})
        return state
    
    # If question failed guardrails, return rejection message
    if not state.get("is_valid"):
        state["response"] = """I appreciate your question, but I'm specifically designed to answer professional queries about Andy Darmawan's experience, skills, and background. 

Your question falls outside my scope. If you have any questions about Andy's professional profile, I'd be happy to help!"""
        state["conversation_history"].append({"role": "user", "content": state["message"]})
        state["conversation_history"].append({"role": "assistant", "content": state["response"]})
        return state
    
    # Build normal chat prompt with context
    if state.get("context"):
        prompt = f"""You are a chatbot representing Andy Darmawan. Answer like you are Andy Darmawan. Your role is to answer professional queries about Andy based on the provided information. Be helpful, professional, and accurate.

Relevant Information:
{state["context"]}

Question: {state["message"]}

Answer:"""
    else:
        prompt = state["message"]
    
    result = chat_model.invoke(prompt)
    state["response"] = result.content
    
    # Record conversation in history
    state["conversation_history"].append({"role": "user", "content": state["message"]})
    state["conversation_history"].append({"role": "assistant", "content": state["response"]})
    
    return state
