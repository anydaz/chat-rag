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
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        streaming=True,
    )
    
    # Initialize conversation history if not present
    if "conversation_history" not in state:
        state["conversation_history"] = []
    
    # If calendar validation failed (is_valid=false, but calendar_action exists)
    if state.get("calendar_action") and not state.get("is_valid"):
        # calendar_result contains the gathered info and missing fields
        calendar_info = state.get("calendar_result", "")
        
        # Use LLM to present this information gracefully
        prompt = f"""You are a helpful calendar assistant. Present the following information to the user in a friendly, conversational way.

Information gathered and what's still needed:
{calendar_info}

User's latest message: {state["message"]}

Respond naturally and ask for the missing information in a helpful tone."""
        
        result = chat_model.invoke(prompt)
        state["response"] = result.content
        state["conversation_history"].append({"role": "user", "content": state["message"]})
        state["conversation_history"].append({"role": "assistant", "content": state["response"]})
        return state
    
    # If calendar action succeeded (is_valid=true and calendar agent ran)
    if state.get("calendar_action") and state.get("calendar_result"):
        calendar_result = state.get("calendar_result", "")
        
        # Use LLM to interpret the calendar agent's result and respond gracefully
        prompt = f"""You are a helpful calendar assistant. Interpret the following calendar operation result and provide a graceful response to the user.

Calendar operation result:
{calendar_result}

User's original request: {state["message"]}

Respond naturally to confirm what was done or explain the result in a friendly tone."""
        
        result = chat_model.invoke(prompt)
        state["response"] = result.content
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
