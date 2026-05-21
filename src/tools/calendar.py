"""Google Calendar toolkit integration for LangGraph agents."""

from langchain_google_community import CalendarToolkit
import os


def get_calendar_toolkit():
    """Initialize and return Google Calendar toolkit.
    
    Expects credentials.json in the project root.
    On first run, will prompt for Google auth and create token.json
    """
    try:
        # CalendarToolkit automatically reads credentials.json
        toolkit = CalendarToolkit()
        print("✅ Google Calendar toolkit initialized successfully")
        return toolkit
        
    except FileNotFoundError as e:
        raise ValueError(
            f"Google Calendar setup error: {e}\n"
            "Make sure credentials.json is in the project root.\n"
            "Get it from: https://developers.google.com/calendar/api/quickstart/python"
        )
    except Exception as e:
        raise ValueError(f"Google Calendar initialization failed: {e}")


def get_calendar_tools():
    """Get all available calendar tools for use in agents."""
    toolkit = get_calendar_toolkit()
    return toolkit.get_tools()
