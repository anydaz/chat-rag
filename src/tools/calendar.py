"""Google Calendar toolkit integration for LangGraph agents."""

from langchain_google_community import CalendarToolkit
from .calendar_auth import GoogleCalendarAuth
import os


def get_calendar_toolkit():
    """Initialize and return Google Calendar toolkit with auto token refresh.
    
    Handles automatic token refresh when tokens expire using stored refresh_token.
    Automatically triggers OAuth2 authentication if token.json is missing.
    
    Returns:
        CalendarToolkit: Initialized toolkit with valid credentials
        
    Raises:
        FileNotFoundError: If credentials.json not found
        RuntimeError: If authentication fails
        Exception: If toolkit initialization fails
    """
    try:
        # Ensure credentials are fresh (auto-refreshes if needed)
        # If token.json is missing, this will trigger OAuth2 flow
        credentials = GoogleCalendarAuth.get_credentials()
        print("✅ Google Calendar toolkit initialized with auto-refresh enabled")
        
        # CalendarToolkit will use the refreshed credentials
        toolkit = CalendarToolkit()
        return toolkit
        
    except FileNotFoundError as e:
        # Missing credentials.json
        raise ValueError(
            f"Google Calendar setup error: {e}\n"
            "Get credentials.json from: https://developers.google.com/calendar/api/quickstart/python"
        )
    except RuntimeError as e:
        # Authentication failed
        raise RuntimeError(f"Google Calendar authentication failed: {e}")
    except Exception as e:
        raise ValueError(f"Google Calendar initialization failed: {e}")


def get_calendar_tools():
    """Get all available calendar tools for use in agents."""
    toolkit = get_calendar_toolkit()
    return toolkit.get_tools()
