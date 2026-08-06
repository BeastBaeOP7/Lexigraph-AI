from langgraph.graph import StateGraph, START, END
from app.graph.state import GraphState
from app.nodes.classify import classify_intent_node
from app.nodes.retrieve import retrieve_node
from app.nodes.confidence import validate_retrieval_node
from app.nodes.generate import generate_answer_node
from app.nodes.citations import enforce_citations_node
from app.nodes.rerank import format_response_node

def check_intent_route(state: GraphState) -> str:
    """Routes the graph based on classified intent."""
    intent = state.get("intent", "UNKNOWN")
    if intent in ["QUESTION", "CLAUSE_SEARCH", "COMPARE", "EXTRACT", "SUMMARIZE"]:
        return intent
    return "UNKNOWN"

def check_validation_route(state: GraphState) -> str:
    """Routes the graph: goes directly to format response if validation failed."""
    if not state.get("validated_chunks"):
        return "format"
    return "generate"

# Setup the state graph
builder = StateGraph(GraphState)

# Add all process nodes
builder.add_node("classify", classify_intent_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("validate", validate_retrieval_node)
builder.add_node("generate", generate_answer_node)
builder.add_node("citations", enforce_citations_node)
builder.add_node("format", format_response_node)

# Add static transition edges
builder.add_edge(START, "classify")

# Add conditional transition edge based on intent classification
builder.add_conditional_edges(
    "classify",
    check_intent_route,
    {
        "QUESTION": "retrieve",
        "CLAUSE_SEARCH": "retrieve",
        "COMPARE": "retrieve",
        "EXTRACT": "retrieve",
        "SUMMARIZE": "retrieve",
        "UNKNOWN": "retrieve"
    }
)

builder.add_edge("retrieve", "validate")

# Add conditional transition edge based on validation results
builder.add_conditional_edges(
    "validate",
    check_validation_route,
    {
        "generate": "generate",
        "format": "format"
    }
)

builder.add_edge("generate", "citations")
builder.add_edge("citations", "format")
builder.add_edge("format", END)

# Compile graph
app_graph = builder.compile()
