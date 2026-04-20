from langgraph.graph import StateGraph, END

from graph.state import PaperState
from graph.nodes import (
    extract_node,
    summarise_node,
    evaluate_score_fit_node,
    evaluate_reason_node,
    route_evaluation_after_score_fit,
    discovery_init_node,
    discovery_search_node,
    discovery_evaluate_node,
    route_discovery_loop,
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


def build_discovery_pipeline():
    """Assemble and compile the topic-only discovery graph."""
    graph = StateGraph(PaperState)

    graph.add_node("discovery_init", discovery_init_node)
    graph.add_node("discovery_search", discovery_search_node)
    graph.add_node("discovery_evaluate", discovery_evaluate_node)

    graph.set_entry_point("discovery_init")
    graph.add_edge("discovery_init", "discovery_search")
    graph.add_edge("discovery_search", "discovery_evaluate")
    graph.add_conditional_edges(
        "discovery_evaluate",
        route_discovery_loop,
        {"search": "discovery_search", "end": END},
    )

    return graph.compile()

# Singleton — compile once and reuse
pipeline = build_pipeline()
discovery_pipeline = build_discovery_pipeline()
