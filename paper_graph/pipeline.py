from langgraph.graph import StateGraph, END

from paper_graph.state import PaperState
from paper_graph.nodes import (
    extract_node,
    summarise_node,
    evaluate_score_fit_node,
    evaluate_reason_node,
    evaluate_matrix_node,
    route_evaluation_after_score_fit,
    route_evaluation_after_reason,
    discovery_init_node,
    discovery_search_node,
    discovery_triage_candidates_node,
    discovery_prepare_candidates_node,
    discovery_pick_candidate_node,
    discovery_evaluate_candidate_node,
    discovery_eval_early_exit_node,
    discovery_source_profile_node,
    discovery_finalize_candidate_node,
    discovery_round_check_node,
    route_discovery_candidate,
    route_after_discovery_evaluate,
    route_after_discovery_finalize,
    route_discovery_loop,
)


def build_pipeline():
    """Assemble and compile the research assistant graph."""
    graph = StateGraph(PaperState)

    graph.add_node("extract", extract_node)
    graph.add_node("summarise", summarise_node)
    graph.add_node("evaluate_score_fit", evaluate_score_fit_node)
    graph.add_node("evaluate_reason", evaluate_reason_node)
    graph.add_node("evaluate_matrix", evaluate_matrix_node)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "summarise")
    graph.add_edge("summarise", "evaluate_score_fit")
    graph.add_conditional_edges(
        "evaluate_score_fit",
        route_evaluation_after_score_fit,
        {"reason": "evaluate_reason", "matrix": "evaluate_matrix", "end": END},
    )
    graph.add_conditional_edges(
        "evaluate_reason",
        route_evaluation_after_reason,
        {"matrix": "evaluate_matrix", "end": END},
    )
    graph.add_edge("evaluate_matrix", END)

    return graph.compile()


def build_discovery_pipeline():
    """Assemble and compile the topic-only discovery graph."""
    graph = StateGraph(PaperState)

    graph.add_node("discovery_init", discovery_init_node)
    graph.add_node("discovery_search", discovery_search_node)
    graph.add_node("discovery_triage_candidates", discovery_triage_candidates_node)
    graph.add_node("discovery_prepare_candidates", discovery_prepare_candidates_node)
    graph.add_node("discovery_pick_candidate", discovery_pick_candidate_node)
    graph.add_node("discovery_evaluate_candidate", discovery_evaluate_candidate_node)
    graph.add_node("discovery_eval_early_exit", discovery_eval_early_exit_node)
    graph.add_node("discovery_source_profile", discovery_source_profile_node)
    graph.add_node("discovery_finalize_candidate", discovery_finalize_candidate_node)
    graph.add_node("discovery_round_check", discovery_round_check_node)

    graph.set_entry_point("discovery_init")
    graph.add_edge("discovery_init", "discovery_search")
    graph.add_edge("discovery_search", "discovery_triage_candidates")
    graph.add_edge("discovery_triage_candidates", "discovery_prepare_candidates")
    graph.add_edge("discovery_prepare_candidates", "discovery_pick_candidate")
    graph.add_conditional_edges(
        "discovery_pick_candidate",
        route_discovery_candidate,
        {
            "evaluate": "discovery_evaluate_candidate",
            "round_check": "discovery_round_check",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "discovery_evaluate_candidate",
        route_after_discovery_evaluate,
        {
            "profile": "discovery_source_profile",
            "early_exit": "discovery_eval_early_exit",
        },
    )
    graph.add_edge("discovery_source_profile", "discovery_finalize_candidate")
    graph.add_edge("discovery_eval_early_exit", "discovery_finalize_candidate")
    graph.add_conditional_edges(
        "discovery_finalize_candidate",
        route_after_discovery_finalize,
        {
            "pick": "discovery_pick_candidate",
            "round_check": "discovery_round_check",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "discovery_round_check",
        route_discovery_loop,
        {"search": "discovery_search", "end": END},
    )

    return graph.compile()

# Singleton — compile once and reuse
pipeline = build_pipeline()
discovery_pipeline = build_discovery_pipeline()
