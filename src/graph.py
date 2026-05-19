"""LangGraph definition for chat with RAG."""

from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from .rag import retrieve_context, load_documents_to_chroma
import os


class ChatState(TypedDict):
    """State for chat messages."""
    message: str
    response: str
    context: str
    is_valid: bool  # Whether the question passed guardrails


def create_chat_graph():
    """Create a chat graph with RAG retrieval and guardrails."""
    
    # Initialize OpenAI chat model with streaming
    chat_model = ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
        streaming=True,
    )
    
    # Initialize guardrails model
    guardrails_model = ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,  # Lower temperature for consistent guardrail decisions
    )
    
    # Load documents to Chroma on initialization
    try:
        load_documents_to_chroma()
    except Exception as e:
        print(f"Warning: Could not load documents to Chroma: {e}")
    
    def guardrails_node(state: ChatState) -> ChatState:
        """Check if the question is relevant to a professional profile."""
        guardrails_prompt = f"""You are a gatekeeping AI for a professional chatbot representing a person. 
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

Questions are NOT RELEVANT if they ask about:
- General knowledge unrelated to the person's profile
- Other people's information
- Inappropriate or harmful content
- Topics completely unrelated to the person's professional profile

User Question: "{state['message']}"

Respond with ONLY "VALID" or "INVALID" - nothing else."""
        
        result = guardrails_model.invoke(guardrails_prompt)
        print("guardrails result:", result.content.strip())
        is_valid = result.content.upper() == "VALID"
        
        state["is_valid"] = is_valid
        if not is_valid:
            print(f"⚠️  Question blocked by guardrails: {state['message']}")
        
        return state
    
    def retrieve_node(state: ChatState) -> ChatState:
        """Retrieve relevant context from documents."""
        context = retrieve_context(state["message"], top_k=3)
        state["context"] = context
        return state
    
    def chat_node(state: ChatState) -> ChatState:
        """Call OpenAI with the user message and context."""
        # If question failed guardrails, return rejection message
        if not state["is_valid"]:
            state["response"] = """I appreciate your question, but I'm specifically designed to answer professional queries about Andy Darmawan's experience, skills, and background. 

Your question falls outside my scope. If you have any questions about Andy's professional profile, I'd be happy to help!"""
            return state
        
        # Build prompt with context
        if state["context"]:
            prompt = f"""You are a chatbot representing Andy Darmawan answer like you are Andy Darmawan. Your role is to answer professional queries about Andy based on the provided information. Be helpful, professional, and accurate.

Relevant Information:
{state["context"]}

Question: {state["message"]}

Answer:"""
        else:
            prompt = state["message"]
        
        result = chat_model.invoke(prompt)
        state["response"] = result.content
        return state
    
    def should_retrieve(state: ChatState) -> str:
        """Router: Determine if question passed guardrails."""
        return "retrieve" if state["is_valid"] else "respond"
    
    # Build the LangGraph
    graph = StateGraph(ChatState)
    
    # Add nodes
    graph.add_node("guardrails", guardrails_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("chat", chat_node)
    
    # Set entry point
    graph.set_entry_point("guardrails")
    
    # Add edges with conditional routing
    graph.add_conditional_edges(
        "guardrails",
        should_retrieve,
        {
            "retrieve": "retrieve",
            "respond": "chat"
        }
    )
    
    # If question is valid, retrieve then chat
    graph.add_edge("retrieve", "chat")
    
    # Chat is the final node
    graph.set_finish_point("chat")
    
    # Compile and return
    return graph.compile()

