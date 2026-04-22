from typing import Any, NotRequired, Optional, TypedDict

class PaperState(TypedDict):
    filename: str
    pdf_text: str
    summary: str
    key_findings: str
    methodology: str
    relevance_score: float
    relevance_reason: str
    source_profile: NotRequired[dict[str, str]]
    fit: bool
    error: Optional[str]
    trace: NotRequired[list[dict[str, Any]]]
    topic: NotRequired[str]
    discovery_query: NotRequired[str]
    discovery_cursor: NotRequired[int]
    discovery_batch_size: NotRequired[int]
    max_discovery_rounds: NotRequired[int]
    discovery_round: NotRequired[int]
    target_qualified_count: NotRequired[int]
    discovered_candidates: NotRequired[list[dict[str, Any]]]
    evaluated_candidates: NotRequired[list[dict[str, Any]]]
    qualified_works: NotRequired[list[dict[str, Any]]]
    candidate_queue: NotRequired[list[dict[str, Any]]]
    current_candidate: NotRequired[dict[str, Any] | None]
    candidate_score: NotRequired[float]
    candidate_fit: NotRequired[bool]
    candidate_quality: NotRequired[bool]
    candidate_reason: NotRequired[str]
    candidate_source_profile: NotRequired[dict[str, str]]
    candidate_eval_duration_ms: NotRequired[float | None]
