from langgraph.graph import StateGraph, END
from graph.state import PaperState
from graph.nodes import extract_node, summarise_node, evaluate_node

def build_pipeline():
    """Assemble and compile the research assistant graph."""
    graph = StateGraph(PaperState)

    graph.add_node("extract", extract_node)
    graph.add_node("summarise", summarise_node)
    graph.add_node("evaluate", evaluate_node)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "summarise")
    graph.add_edge("summarise", "evaluate")
    graph.add_edge("evaluate", END)

    return graph.compile()

# Singleton — compile once and reuse
pipeline = build_pipeline()
