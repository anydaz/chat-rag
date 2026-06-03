"""Document retrieval node."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..graph import ChatState

from ..rag import retrieve_context


def retrieve_node(state: "ChatState") -> "ChatState":
    """Retrieve relevant context from documents."""
    context = retrieve_context(state["message"], top_k=3)
    state["context"] = context
    return state
