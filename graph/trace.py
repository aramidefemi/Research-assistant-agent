"""Append-only pipeline trace steps for multi-node (agent) visibility."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from graph.state import PaperState

# Human-readable labels for trace `node` ids (UI + flowcharts).
TRACE_NODE_LABELS: dict[str, str] = {
    "extract": "Extract text from PDF",
    "summarise": "Summarise paper",
    "evaluate_score_fit": "Score relevance and fit",
    "evaluate_reason": "Evaluation reasoning",
    "evaluate_matrix": "Evidence matrix",
    "discovery_init": "Set up discovery",
    "discovery_search": "Search for papers",
    "discovery_triage_candidates": "Rank and filter candidates",
    "discovery_prepare_candidates": "Build candidate queue",
    "discovery_pick_candidate": "Select next candidate",
    "discovery_evaluate_candidate": "Evaluate candidate",
    "discovery_score_fit": "Score fit (candidate)",
    "discovery_quality_reason": "Quality and rationale",
    "discovery_eval_early_exit": "Skip source profiling",
    "discovery_source_profile": "Extract source profile",
    "discovery_finalize_candidate": "Record candidate outcome",
    "discovery_round_check": "Check round limits",
    "runtime": "Pipeline error",
}


def trace_step_title(node: str) -> str:
    key = (node or "").strip() or "?"
    if key == "?":
        return "Unknown step"
    return TRACE_NODE_LABELS.get(key, key.replace("_", " ").title())


def append_trace(
    state: PaperState,
    node: str,
    contribution: str,
    *,
    detail: str = "",
    duration_ms: float | None = None,
    result: dict[str, Any] | None = None,
) -> PaperState:
    step: dict[str, Any] = {
        "node": node,
        "contribution": contribution,
        "detail": detail.strip(),
        "duration_ms": duration_ms,
        "at": datetime.now(timezone.utc).isoformat(),
        "result": dict(result or {}),
    }
    prev = list(state.get("trace") or [])
    return {**state, "trace": prev + [step]}
