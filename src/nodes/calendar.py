"""Calendar operations node."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..graph import ChatState

from ..agents.calendar_agent import run_calendar_request


def calendar_node(state: "ChatState") -> "ChatState":
    """Execute calendar operations using ReAct agent after validation.
    
    At this point, all required fields should be present in calendar_data:
    - event_title
    - date_time
    - timezone
    - email_address
    """
    if not state.get("calendar_action"):
        state["calendar_result"] = None
        return state
    
    # If validation didn't pass, don't proceed
    if not state.get("is_valid"):
        print(f"⚠️  Skipping calendar agent - validation incomplete")
        return state
    
    try:
        # Get calendar data extracted from validation
        calendar_data = state.get("calendar_data", {})
        
        # Log calendar intent with extracted data
        print(f"📅 Processing validated calendar request")
        print(f"   Event title: {calendar_data.get('event_title', 'N/A')}")
        print(f"   Date/Time: {calendar_data.get('date_time', 'N/A')}")
        print(f"   Timezone: {calendar_data.get('timezone', 'N/A')}")
        print(f"   Email: {calendar_data.get('email_address', 'N/A')}")
        
        # Run calendar request through ReAct agent with calendar data
        # Pass structured data to the agent
        request_details = f"""
        Create a calendar event with the following details:
        - Event Title: {calendar_data.get('event_title', '')}
        - Date/Time: {calendar_data.get('date_time', '')}
        - Timezone: {calendar_data.get('timezone', '')}
        - Email: {calendar_data.get('email_address', '')}
        """
        
        result = run_calendar_request(request_details)
        state["calendar_result"] = result
        print(f"✅ Calendar agent completed")
        print(f"Result: {result}")
    except Exception as e:
        state["calendar_result"] = f"Error: {str(e)}"
        print(f"❌ Calendar agent error: {e}")
    
    return state
