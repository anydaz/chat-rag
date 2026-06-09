"""Calendar event validation node."""

from typing import TYPE_CHECKING, List
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..graph import ChatState


class ValidationResponse(BaseModel):
    """Schema for calendar event validation response."""
    complete: bool = Field(description="Whether all required fields are present")
    missing_fields: List[str] = Field(default_factory=list, description="List of missing required fields")
    event_title: str = Field(default="", description="Event title/name extracted from conversation")
    date_time: str = Field(default="", description="Date and/or time extracted from conversation")
    timezone: str = Field(default="", description="Timezone extracted from conversation")
    email_address: str = Field(default="", description="Email address extracted from conversation")


VALIDATION_PROMPT = """
Previous conversation history:
{context}

==============

You are a calendar event validator. Your job is to determine if all required information is present in the conversation for creating a calendar event, and extract any information that has been gathered.

Required fields to check for:
1. Event title/name - What is the name or subject of the event?
2. Date and/or time - When should the event happen?
3. Timezone - What timezone should the event be in? Can be timezone name (e.g., America/New_York, UTC, PST, EST, Europe/London), timezone abbreviation (EST, PST, GMT, etc.), or city name (e.g., Jakarta, Bangkok, Singapore, London, New York, Tokyo, etc.)
4. Email address - What is an email address (yours or attendee's)?


Task:
1. Carefully analyze the ENTIRE conversation history above
2. Extract any information found for each field:
   - Event title/name: Look for words describing what the meeting/event is about
   - Date/time: Look for dates, times, days of week, or time references
   - Timezone: Look for timezone names, abbreviations, city names, or timezone references (e.g., "UTC", "PST", "Jakarta", "New York time", "Asia/Jakarta", "EST", etc.)
   - Email: Look for anything that looks like an email address (text@domain.com)
3. Determine which fields are present and which are missing. Make sure it doesn't contradict with each other, for example, the date is given, but we still flag it as missing.
4. Return the extracted information and completion status

For any field not found, return an empty string.
Be thorough and check the entire conversation, not just recent messages."""


def validate_calendar_event(state: "ChatState") -> "ChatState":
    """
    Validate calendar event fields using LLM with structured output.
    
    Checks both current message AND conversation history to determine if all required
    fields are present: event title, date/time, timezone, and email address.
    """
    message = state.get("message", "").strip()
    
    if not message:
        state["calendar_result"] = "Please provide event details (title, date, time, timezone, and your email)"
        return state
    
    # Build context from current message + full conversation history
    full_context = f"Current message from user:\n{message}"
    
    if state.get("conversation_history"):
        # Include full conversation history for context
        history = state["conversation_history"]
        if history:
            full_context += "\n\nFull conversation history:\n"
            for msg in history:
                if isinstance(msg, dict):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    speaker = "User" if role == "user" else "Assistant"
                    full_context += f"{speaker}: {content}\n"
    
    # Use LLM with structured output
    from langchain_openai import ChatOpenAI
    import os
    
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
        )
        
        # Use structured output with Pydantic schema
        structured_llm = llm.with_structured_output(ValidationResponse)
        
        validation_prompt = VALIDATION_PROMPT.format(context=full_context)
        validation_result = structured_llm.invoke(validation_prompt)
        
        is_complete = validation_result.complete
        missing_fields = validation_result.missing_fields
        
        # Log what information has been gathered
        gathered_info = []
        if validation_result.event_title:
            gathered_info.append(f"Event title: {validation_result.event_title}")
        if validation_result.date_time:
            gathered_info.append(f"Date/Time: {validation_result.date_time}")
        if validation_result.timezone:
            gathered_info.append(f"Timezone: {validation_result.timezone}")
        if validation_result.email_address:
            gathered_info.append(f"Email: {validation_result.email_address}")
        
        if gathered_info:
            print(f"📋 Information gathered so far:")
            for info in gathered_info:
                print(f"   - {info}")
        
        if not is_complete:
            missing_str = ", ".join(missing_fields)
            gathered_str = "\n".join(gathered_info) if gathered_info else "No information gathered yet."
            state["calendar_result"] = f"Information gathered so far:\n{gathered_str}\n\nStill needed:\n- {missing_str}\n\nPlease provide the missing information."
            # Store extracted data even if incomplete
            state["calendar_data"] = {
                "event_title": validation_result.event_title,
                "date_time": validation_result.date_time,
                "timezone": validation_result.timezone,
                "email_address": validation_result.email_address,
            }
            print(f"⚠️  Calendar validation incomplete. Missing: {missing_str}")
            return state
        
        # All fields present - mark as valid for calendar agent
        state["is_valid"] = True
        state["calendar_result"] = None  # Clear result since validation passed
        # Store complete extracted data
        state["calendar_data"] = {
            "event_title": validation_result.event_title,
            "date_time": validation_result.date_time,
            "timezone": validation_result.timezone,
            "email_address": validation_result.email_address,
        }
        gathered_str = "\n".join(gathered_info) if gathered_info else "No information."
        print(f"✅ Calendar event validation passed")
        print(f"All information gathered:\n{gathered_str}")
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        state["calendar_result"] = f"Error validating calendar event: {str(e)}"
    
    return state
