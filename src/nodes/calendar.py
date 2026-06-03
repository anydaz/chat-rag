"""Calendar operations node."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..graph import ChatState

from ..agents.calendar_agent import run_calendar_request


def calendar_node(state: "ChatState") -> "ChatState":
    """Execute calendar operations using ReAct agent."""
    if not state.get("calendar_action"):
        state["calendar_result"] = None
        return state
    
    try:
        # Run calendar request through ReAct agent
        result = run_calendar_request(state["message"])
        state["calendar_result"] = result
        print(f"✅ Calendar agent completed")
    except Exception as e:
        state["calendar_result"] = f"Error: {str(e)}"
        print(f"❌ Calendar agent error: {e}")
    
    return state
