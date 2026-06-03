"""Guardrails node for validating user questions."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..graph import ChatState


GUARDRAILS_SYSTEM_PROMPT = """You are a gatekeeping AI for a professional chatbot representing a person. 
Your role is to determine if a user's question is relevant to that person's professional profile and experience.

Questions are RELEVANT if they ask about:
- who the person is and what they do
- skills, experience, or background
- professional accomplishments
- education or certifications
- availability or contact information
- Technical expertise or projects
- Anything related to professional services
- work history or previous roles
- Follow-ups to previous valid questions about the person's profile
- Scheduling meetings or calendar events with the person

Questions are NOT RELEVANT if they ask about:
- General knowledge unrelated to the person's profile
- Other people's information
- Inappropriate or harmful content
- Topics completely unrelated to the person's professional profile

Respond with ONLY "VALID" or "INVALID" - nothing else."""


def guardrails_node(state: "ChatState", guardrails_model) -> "ChatState":
    """Check if the question is relevant to a professional profile."""
    # Build conversation context
    history_text = ""
    if state.get("conversation_history"):
        history_text = "Conversation History:\n"
        for msg in state["conversation_history"][-4:]:
            history_text += f"- {msg['role']}: {msg['content']}\n"
        history_text += "\n"
    
    guardrails_prompt = f"""{GUARDRAILS_SYSTEM_PROMPT}

{history_text}

Current User Question: "{state['message']}"

Respond with ONLY "VALID" or "INVALID" - nothing else."""
    
    result = guardrails_model.invoke(guardrails_prompt)
    print("guardrails result:", result.content.strip())
    is_valid = result.content.upper() == "VALID"
    
    state["is_valid"] = is_valid
    if not is_valid:
        print(f"⚠️  Question blocked by guardrails: {state['message']}")
    
    return state
