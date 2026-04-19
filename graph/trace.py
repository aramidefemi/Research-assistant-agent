"""Append-only pipeline trace steps for multi-node (agent) visibility."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from graph.state import PaperState


def append_trace(
    state: PaperState,
    node: str,
    contribution: str,
    *,
    detail: str = "",
    duration_ms: float | None = None,
) -> PaperState:
    step: dict[str, Any] = {
        "node": node,
        "contribution": contribution,
        "detail": detail.strip(),
        "duration_ms": duration_ms,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    prev = list(state.get("trace") or [])
    return {**state, "trace": prev + [step]}
