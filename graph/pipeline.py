from langgraph.graph import StateGraph, END

from graph.state import PaperState
from graph.nodes import (
    extract_node,
    summarise_node,
    evaluate_score_fit_node,
    evaluate_reason_node,
    route_evaluation_after_score_fit,
)


def build_pipeline():
    """Assemble and compile the research assistant graph."""
    graph = StateGraph(PaperState)

    graph.add_node("extract", extract_node)
    graph.add_node("summarise", summarise_node)
    graph.add_node("evaluate_score_fit", evaluate_score_fit_node)
    graph.add_node("evaluate_reason", evaluate_reason_node)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "summarise")
    graph.add_edge("summarise", "evaluate_score_fit")
    graph.add_conditional_edges(
        "evaluate_score_fit",
        route_evaluation_after_score_fit,
        {"reason": "evaluate_reason", "end": END},
    )
    graph.add_edge("evaluate_reason", END)

    return graph.compile()

# Singleton — compile once and reuse
pipeline = build_pipeline()
