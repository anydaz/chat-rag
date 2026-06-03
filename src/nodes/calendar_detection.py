"""Calendar intent detection node."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..graph import ChatState

from ..agents.calendar.prompts import CALENDAR_INTENT_PROMPT


def detect_calendar_intent(state: "ChatState", guardrails_model) -> "ChatState":
    """Detect if user wants to create calendar events."""
    # Don't check is_valid here - we run FIRST, before guardrails
    
    intent_prompt = CALENDAR_INTENT_PROMPT.format(message=state["message"])
    
    result = guardrails_model.invoke(intent_prompt)
    intent = result.content.strip().upper()
    
    if intent == "CREATE":
        state["calendar_action"] = "CREATE"
        print(f"📅 Calendar intent detected: CREATE")
    else:
        state["calendar_action"] = None
    
    return state
