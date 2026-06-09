"""ReAct agent for calendar operations."""

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from ..tools.calendar import get_calendar_tools
from .calendar.prompts import CALENDAR_SYSTEM_PROMPT
import os


def create_calendar_agent():
    """Create a ReAct agent for calendar operations with field validation."""
    
    # Initialize LLM with strict temperature for consistent behavior
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )
    
    # Bind system prompt to enforce field requirements
    llm_with_system = llm.bind(
        system_prompt=CALENDAR_SYSTEM_PROMPT
    )
    
    # Get calendar tools
    tools = get_calendar_tools()
    
    # Create ReAct agent using LangGraph
    agent = create_react_agent(
        model=llm_with_system,
        tools=tools
    )
    
    return agent


def run_calendar_request(request: str) -> str:
    """Execute a calendar request using the ReAct agent.
    
    Args:
        request: User's calendar request (e.g., "Schedule a meeting on May 25 at 2pm")
    
    Returns:
        Agent's response with action taken
    """
    try:
        agent = create_calendar_agent()
        
        # Invoke the agent with the request
        result = agent.invoke({"messages": [{"role": "user", "content": request}]})
        
        # Extract the response from the agent result
        if isinstance(result, dict) and "messages" in result:
            # Get the last message (should be from assistant)
            messages = result["messages"]
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "content"):
                    return last_msg.content
                elif isinstance(last_msg, dict):
                    return last_msg.get("content", "Calendar operation completed")
        
        return "Calendar operation completed"
    except Exception as e:
        print(f"❌ Calendar agent error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error processing calendar request: {str(e)}"
