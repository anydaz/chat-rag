"""Calendar intent detection node."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..graph import ChatState

from ..agents.calendar.prompts import CALENDAR_INTENT_PROMPT


def detect_calendar_intent(state: "ChatState", guardrails_model) -> "ChatState":
    """Detect if user wants to create calendar events.
    
    Checks both current message AND conversation history for calendar intent.
    """
    # Combine current message with conversation history for better context
    message = state.get("message", "").strip()
    full_context = message
    
    if state.get("conversation_history"):
        # Add recent conversation context (last 10 messages for more context)
        history = state["conversation_history"][-10:]
        for msg in history:
            if isinstance(msg, dict):
                full_context += " " + msg.get("content", "")
    
    intent_prompt = CALENDAR_INTENT_PROMPT.format(message=full_context)
    
    result = guardrails_model.invoke(intent_prompt)
    intent = result.content.strip().upper()
    
    if intent == "CREATE":
        state["calendar_action"] = "CREATE"
        print(f"📅 Calendar intent detected: CREATE")
    else:
        state["calendar_action"] = None
    
    return state
