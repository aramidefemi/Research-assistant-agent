import hashlib
import json
import threading
import time

import streamlit as st

from paper_graph.state import PaperState
from paper_graph.trace import append_trace
from typing import Any, Literal

from utils.prompts import (
    SUMMARISE_PROMPT,
    EVALUATE_SCORE_FIT_PROMPT,
    EVALUATE_REASON_PROMPT,
    EVALUATION_MATRIX_PROMPT,
    DISCOVERY_EVALUATE_CANDIDATE_PROMPT,
    DISCOVERY_SOURCE_PROFILE_PROMPT,
    DISCOVERY_ABSTRACT_TRIAGE_PROMPT,
)
from utils.gemini_llm import invoke_gemini_prompt
from utils.journal_search import search_journals

# Skip LLM source-profile extraction when topical fit + score already imply high confidence.
DISCOVERY_PROFILE_RULE_SCORE_MIN = 0.65

# After scoring: skip matrix/source-profile step for clear rejects (saves rule + LLM work).
DISCOVERY_EVAL_SKIP_PROFILE_BELOW_SCORE = 0.40

# Bump when the discovery-eval prompt or parsing contract changes (invalidates cache entries).
DISCOVERY_EVAL_CACHE_VERSION = 1
DISCOVERY_EVAL_CACHE_MAX = 384
_DISCOVERY_EVAL_CACHE_FALLBACK: dict[str, dict[str, Any]] = {}
_DISCOVERY_EVAL_CACHE_FALLBACK_LOCK = threading.Lock()


def _discovery_eval_cache_bucket() -> dict[str, dict[str, Any]]:
    try:
        k = "_discovery_eval_cache_v1"
        if k not in st.session_state:
            st.session_state[k] = {}
        return st.session_state[k]  # type: ignore[return-value]
    except Exception:
        return _DISCOVERY_EVAL_CACHE_FALLBACK


def _discovery_eval_cache_key(topic: str, candidate: dict[str, Any]) -> str:
    doi = str(candidate.get("doi") or "")
    url = str(candidate.get("url") or "")
    title = str(candidate.get("title") or "")
    abstract = str(candidate.get("abstract") or "")[:4000]
    venue = str(candidate.get("venue") or "")
    year = str(candidate.get("year") or "")
    cited = str(candidate.get("cited_by_count") or "")
    raw = (
        f"v{DISCOVERY_EVAL_CACHE_VERSION}\0{topic}\0{doi}\0{url}\0{title}\0"
        f"{venue}\0{year}\0{cited}\0{abstract}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _discovery_eval_cache_get(key: str) -> dict[str, Any] | None:
    bucket = _discovery_eval_cache_bucket()
    if bucket is _DISCOVERY_EVAL_CACHE_FALLBACK:
        with _DISCOVERY_EVAL_CACHE_FALLBACK_LOCK:
            return bucket.get(key)
    return bucket.get(key)


def _discovery_eval_cache_set(key: str, payload: dict[str, Any]) -> None:
    bucket = _discovery_eval_cache_bucket()
    if bucket is _DISCOVERY_EVAL_CACHE_FALLBACK:
        with _DISCOVERY_EVAL_CACHE_FALLBACK_LOCK:
            bucket[key] = payload
            while len(bucket) > DISCOVERY_EVAL_CACHE_MAX:
                bucket.pop(next(iter(bucket)))
        return
    bucket[key] = payload
    while len(bucket) > DISCOVERY_EVAL_CACHE_MAX:
        bucket.pop(next(iter(bucket)))


def extract_node(state: PaperState) -> PaperState:
    """Node 1: Text is already extracted before the graph runs — just validate."""
    text = (state.get("pdf_text") or "").strip()
    if len(text) < 100:
        return append_trace(
            {**state, "error": "Insufficient text extracted from PDF."},
            "extract",
            "Stopped pipeline — unusable PDF text.",
            detail="Fewer than 100 non-whitespace characters.",
            result={"status": "error", "chars_extracted": len(text)},
        )
    return append_trace(
        state,
        "extract",
        "Validated corpus; handed off to summarisation.",
        detail=f"{len(text):,} characters of extracted text.",
        result={"status": "ok", "chars_extracted": len(text)},
    )


def summarise_node(state: PaperState) -> PaperState:
    """Node 2: Summarise the paper and extract key info."""
    if state.get("error"):
        return state

    try:
        prompt = SUMMARISE_PROMPT.format(pdf_text=state["pdf_text"])
        t0 = time.perf_counter()
        content = invoke_gemini_prompt(prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Parse structured response
        summary = _extract_section(content, "SUMMARY:", "KEY_FINDINGS:")
        key_findings = _extract_section(content, "KEY_FINDINGS:", "METHODOLOGY:")
        methodology = _extract_section(content, "METHODOLOGY:", None)

        s2 = {
            **state,
            "summary": summary.strip(),
            "key_findings": key_findings.strip(),
            "methodology": methodology.strip(),
        }
        return append_trace(
            s2,
            "summarise",
            "Structured summary, findings, and methodology from the full text.",
            detail=f"Gemini call completed in {elapsed_ms:.0f} ms.",
            duration_ms=elapsed_ms,
            result={
                "summary": summary.strip(),
                "key_findings": key_findings.strip(),
                "methodology": methodology.strip(),
            },
        )
    except Exception as e:
        return append_trace(
            {**state, "error": f"Summarisation failed: {str(e)}"},
            "summarise",
            "Summarisation agent failed.",
            detail=str(e),
        )


def _parse_score_fit(content: str) -> tuple[float, bool]:
    score = 0.0
    fit = False
    for line in content.split("\n"):
        if line.startswith("SCORE:"):
            try:
                score = float(line.replace("SCORE:", "").strip())
            except ValueError:
                pass
        elif line.startswith("FIT:"):
            fit = "YES" in line.upper()
    return score, fit


def _parse_reason(content: str) -> str:
    for line in content.split("\n"):
        if line.startswith("REASON:"):
            return line.replace("REASON:", "").strip()
    return "Could not parse evaluation."


def evaluate_score_fit_node(state: PaperState) -> PaperState:
    """Score and fit vs research focus (first evaluation step)."""
    if state.get("error"):
        return state

    try:
        research_focus = st.session_state.get("research_focus", "General ML and AI research")
        prompt = EVALUATE_SCORE_FIT_PROMPT.format(
            research_focus=research_focus,
            summary=state["summary"],
            key_findings=state["key_findings"],
            methodology=state["methodology"],
        )
        t0 = time.perf_counter()
        content = invoke_gemini_prompt(prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        score, fit = _parse_score_fit(content)
        quick = st.session_state.get("evaluation_depth", "full") == "quick"
        reason = (
            "Quick evaluation: narrative reasoning skipped."
            if quick
            else ""
        )
        ev: PaperState = {
            **state,
            "relevance_score": score,
            "fit": fit,
            "relevance_reason": reason,
        }
        fit_lbl = "YES" if fit else "NO"
        detail = f"model call {elapsed_ms:.0f} ms."
        if quick:
            detail = f"{detail} (quick mode — no separate reason step)."
        return append_trace(
            ev,
            "evaluate_score_fit",
            f"Relevance scoring vs your focus: SCORE={score:.2f}, FIT={fit_lbl}.",
            detail=detail,
            duration_ms=elapsed_ms,
            result={
                "score": score,
                "fit": fit,
                "mode": "quick" if quick else "full",
            },
        )
    except Exception as e:
        return append_trace(
            {**state, "error": f"Evaluation failed: {str(e)}"},
            "evaluate_score_fit",
            "Score/fit evaluation failed.",
            detail=str(e),
        )


def evaluate_reason_node(state: PaperState) -> PaperState:
    """Narrative justification (second evaluation step; optional via routing)."""
    if state.get("error"):
        return state

    try:
        research_focus = st.session_state.get("research_focus", "General ML and AI research")
        score = state["relevance_score"]
        fit = state["fit"]
        fit_label = "YES" if fit else "NO"
        prompt = EVALUATE_REASON_PROMPT.format(
            research_focus=research_focus,
            summary=state["summary"],
            key_findings=state["key_findings"],
            methodology=state["methodology"],
            score=score,
            fit_label=fit_label,
        )
        t0 = time.perf_counter()
        content = invoke_gemini_prompt(prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        reason = _parse_reason(content)
        ev: PaperState = {**state, "relevance_reason": reason}
        excerpt = reason[:280] + ("…" if len(reason) > 280 else "")
        return append_trace(
            ev,
            "evaluate_reason",
            "Narrative explanation for the score and fit.",
            detail=f"{excerpt} | model call {elapsed_ms:.0f} ms.",
            duration_ms=elapsed_ms,
            result={"reason": reason},
        )
    except Exception as e:
        return append_trace(
            {**state, "error": f"Evaluation reasoning failed: {str(e)}"},
            "evaluate_reason",
            "Reasoning step failed.",
            detail=str(e),
        )


def evaluate_matrix_node(state: PaperState) -> PaperState:
    """Structured evidence matrix extraction (dedicated evaluation step)."""
    if state.get("error"):
        return state

    try:
        research_focus = st.session_state.get("research_focus", "General ML and AI research")
        prompt = EVALUATION_MATRIX_PROMPT.format(
            research_focus=research_focus,
            summary=state.get("summary", ""),
            key_findings=state.get("key_findings", ""),
            methodology=state.get("methodology", ""),
            pdf_text_excerpt=_build_pdf_excerpt(state.get("pdf_text", "")),
        )
        t0 = time.perf_counter()
        content = invoke_gemini_prompt(prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        source_profile = _parse_source_profile(content)
        s2: PaperState = {
            **state,
            "source_profile": source_profile,
        }
        return append_trace(
            s2,
            "evaluate_matrix",
            "Extracted dedicated evaluation matrix fields from paper content.",
            duration_ms=elapsed_ms,
            result={"source_profile": source_profile},
        )
    except Exception as e:
        return append_trace(
            {**state, "error": f"Evaluation matrix extraction failed: {str(e)}"},
            "evaluate_matrix",
            "Evaluation matrix extraction failed.",
            detail=str(e),
        )


def route_evaluation_after_score_fit(state: PaperState) -> Literal["reason", "matrix", "end"]:
    """Orchestration: full evaluation runs a second LLM for REASON; quick stops after score/fit."""
    if state.get("error"):
        return "end"
    if not bool(state.get("fit")):
        return "end"
    if st.session_state.get("evaluation_depth", "full") == "quick":
        return "matrix"
    return "reason"


def route_evaluation_after_reason(state: PaperState) -> Literal["matrix", "end"]:
    """Continue to matrix extraction after narrative reasoning."""
    if state.get("error"):
        return "end"
    if not bool(state.get("fit")):
        return "end"
    return "matrix"


def discovery_init_node(state: PaperState) -> PaperState:
    """Initialize discovery loop state for topic-only flow."""
    topic = (state.get("topic") or "").strip()
    if not topic:
        return append_trace(
            {**state, "error": "Topic is required for journal discovery."},
            "discovery_init",
            "Stopped discovery — missing topic.",
        )
    s2: PaperState = {
        **state,
        "discovery_query": topic,
        "discovery_cursor": int(state.get("discovery_cursor") or 1),
        "discovery_batch_size": int(state.get("discovery_batch_size") or 8),
        "max_discovery_rounds": int(state.get("max_discovery_rounds") or 5),
        "discovery_round": int(state.get("discovery_round") or 0),
        "target_qualified_count": int(state.get("target_qualified_count") or 2),
        "discovered_candidates": list(state.get("discovered_candidates") or []),
        "evaluated_candidates": list(state.get("evaluated_candidates") or []),
        "qualified_works": list(state.get("qualified_works") or []),
    }
    return append_trace(
        s2,
        "discovery_init",
        "Initialized strict orchestrator for topic-only journal discovery.",
        detail=f"target={s2['target_qualified_count']} | max_rounds={s2['max_discovery_rounds']}",
        result={
            "topic": topic,
            "target_qualified_count": s2["target_qualified_count"],
            "max_discovery_rounds": s2["max_discovery_rounds"],
            "discovery_batch_size": s2["discovery_batch_size"],
        },
    )


def discovery_search_node(state: PaperState) -> PaperState:
    """Fetch one batch of journal candidates from external search."""
    if state.get("error"):
        return state

    topic = state.get("discovery_query") or state.get("topic") or ""
    page = int(state.get("discovery_cursor") or 1)
    batch_size = int(state.get("discovery_batch_size") or 8)
    try:
        candidates = search_journals(topic, page=page, per_page=batch_size)
        s2: PaperState = {
            **state,
            "discovered_candidates": candidates,
            "discovery_cursor": page + 1,
            "discovery_round": int(state.get("discovery_round") or 0) + 1,
        }
        return append_trace(
            s2,
            "discovery_search",
            f"Discovered {len(candidates)} candidate journal works for topic search.",
            detail=f"query='{topic}' | page={page}",
            result={
                "query": topic,
                "page": page,
                "count": len(candidates),
                "candidates": candidates,
            },
        )
    except Exception as e:
        return append_trace(
            {**state, "error": f"Journal discovery failed: {str(e)}"},
            "discovery_search",
            "External journal search failed.",
            detail=str(e),
        )


def discovery_prepare_candidates_node(state: PaperState) -> PaperState:
    """Prepare queue for one-candidate-at-a-time evaluation."""
    if state.get("error"):
        return state
    candidates = list(state.get("discovered_candidates") or [])
    s2: PaperState = {
        **state,
        "candidate_queue": candidates,
        "current_candidate": None,
    }
    return append_trace(
        s2,
        "discovery_prepare_candidates",
        "Prepared candidate queue for strict per-candidate evaluation nodes.",
        detail=f"queued={len(candidates)}",
        result={"queued_count": len(candidates), "queue": candidates},
    )


def discovery_triage_candidates_node(state: PaperState) -> PaperState:
    """Rank current round candidates by abstract-only AI triage."""
    if state.get("error"):
        return state
    candidates = list(state.get("discovered_candidates") or [])
    if not candidates:
        return append_trace(
            state,
            "discovery_triage_candidates",
            "No candidates to triage in this round.",
            result={"ranked_count": 0, "refetch": True},
        )

    topic = (state.get("topic") or "").strip()
    try:
        candidates_block = _build_candidate_triage_block(candidates)
        prompt = DISCOVERY_ABSTRACT_TRIAGE_PROMPT.format(
            topic=topic,
            candidates_block=candidates_block,
        )
        t0 = time.perf_counter()
        content = invoke_gemini_prompt(prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ranked_indices, refetch, reason = _parse_discovery_rank(content)
        if refetch or not ranked_indices:
            s2: PaperState = {**state, "discovered_candidates": []}
            return append_trace(
                s2,
                "discovery_triage_candidates",
                "Abstract triage found no promising candidates; requesting next offset batch.",
                detail=reason[:200],
                duration_ms=elapsed_ms,
                result={"ranked_count": 0, "refetch": True, "reason": reason},
            )

        valid_ranked_indices = [idx for idx in ranked_indices if 0 <= idx < len(candidates)]
        if not valid_ranked_indices:
            s2: PaperState = {**state, "discovered_candidates": []}
            return append_trace(
                s2,
                "discovery_triage_candidates",
                "Abstract triage returned no valid ranked candidates; requesting next offset batch.",
                detail=reason[:200],
                duration_ms=elapsed_ms,
                result={"ranked_count": 0, "refetch": True, "reason": reason},
            )
        ranked_candidates = [candidates[idx] for idx in valid_ranked_indices]
        s2 = {**state, "discovered_candidates": ranked_candidates}
        return append_trace(
            s2,
            "discovery_triage_candidates",
            "Ranked candidates by abstract-only AI triage before evaluation.",
            detail=reason[:200],
            duration_ms=elapsed_ms,
            result={
                "ranked_count": len(ranked_candidates),
                "refetch": False,
                "ranked_indices": valid_ranked_indices,
                "reason": reason,
            },
        )
    except Exception as e:
        return append_trace(
            state,
            "discovery_triage_candidates",
            "Abstract triage failed; keeping search order for evaluation.",
            detail=str(e),
            result={"ranked_count": len(candidates), "fallback": "search_order"},
        )


def discovery_pick_candidate_node(state: PaperState) -> PaperState:
    """Pick next candidate from prepared queue."""
    if state.get("error"):
        return state
    queue = list(state.get("candidate_queue") or [])
    if not queue:
        return append_trace(
            {**state, "current_candidate": None, "candidate_queue": queue},
            "discovery_pick_candidate",
            "No more candidates in current round queue.",
            result={"picked": None, "remaining_queue": len(queue)},
        )
    current = queue.pop(0)
    s2 = {
        **state,
        "candidate_queue": queue,
        "current_candidate": current,
    }
    return append_trace(
        s2,
        "discovery_pick_candidate",
        "Picked next candidate from triaged queue for evaluation.",
        detail=current.get("title", "")[:120],
        result={"picked": current, "remaining_queue": len(queue)},
    )


def discovery_evaluate_candidate_node(state: PaperState) -> PaperState:
    """Single LLM pass: relevance (score/fit) + quality + reason."""
    if state.get("error"):
        return state

    topic = (state.get("topic") or "").strip()
    candidate = state.get("current_candidate")
    if not candidate:
        return state

    cache_key = _discovery_eval_cache_key(topic, candidate)
    cached = _discovery_eval_cache_get(cache_key)
    if cached:
        score = float(cached["score"])
        fit = bool(cached["fit"])
        quality = bool(cached["quality"])
        reason = str(cached["reason"])
        elapsed_ms = float(cached.get("duration_ms") or 0.0)
        s2: PaperState = {
            **state,
            "candidate_score": score,
            "candidate_fit": fit,
            "candidate_quality": quality,
            "candidate_reason": reason,
            "candidate_eval_duration_ms": elapsed_ms,
        }
        return append_trace(
            s2,
            "discovery_evaluate_candidate",
            f"Evaluated candidate (cached): SCORE={score:.2f}, FIT={'YES' if fit else 'NO'}, "
            f"QUALITY={'YES' if quality else 'NO'}.",
            detail=reason[:200],
            duration_ms=elapsed_ms,
            result={
                "candidate_title": candidate.get("title", ""),
                "score": score,
                "fit": fit,
                "quality": quality,
                "reason": reason,
                "cache_hit": True,
            },
        )

    try:
        prompt = DISCOVERY_EVALUATE_CANDIDATE_PROMPT.format(
            topic=topic,
            title=candidate.get("title", ""),
            abstract=candidate.get("abstract", ""),
            venue=candidate.get("venue", ""),
            year=candidate.get("year", ""),
            cited_by_count=candidate.get("cited_by_count", 0),
        )
        t0 = time.perf_counter()
        content = invoke_gemini_prompt(prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        score, fit = _parse_score_fit(content)
        quality = _parse_yes_no(content, "QUALITY:")
        reason = _parse_reason(content)
        _discovery_eval_cache_set(
            cache_key,
            {
                "score": score,
                "fit": fit,
                "quality": quality,
                "reason": reason,
                "duration_ms": elapsed_ms,
            },
        )
        s2 = {
            **state,
            "candidate_score": score,
            "candidate_fit": fit,
            "candidate_quality": quality,
            "candidate_reason": reason,
            "candidate_eval_duration_ms": elapsed_ms,
        }
        return append_trace(
            s2,
            "discovery_evaluate_candidate",
            f"Evaluated candidate: SCORE={score:.2f}, FIT={'YES' if fit else 'NO'}, "
            f"QUALITY={'YES' if quality else 'NO'}.",
            detail=reason[:200],
            duration_ms=elapsed_ms,
            result={
                "candidate_title": candidate.get("title", ""),
                "score": score,
                "fit": fit,
                "quality": quality,
                "reason": reason,
                "cache_hit": False,
            },
        )
    except Exception as e:
        return append_trace(
            {**state, "error": f"Discovery candidate evaluation failed: {str(e)}"},
            "discovery_evaluate_candidate",
            "Candidate evaluation failed.",
            detail=str(e),
        )


def _discovery_skip_source_profile_after_eval(state: PaperState) -> tuple[bool, str]:
    """Whether to bypass source_profile entirely based on scoring-only thresholds."""
    if state.get("error"):
        return False, ""
    fit = bool(state.get("candidate_fit"))
    score = float(state.get("candidate_score") or 0.0)
    if not fit:
        return True, "FIT=NO"
    if score < DISCOVERY_EVAL_SKIP_PROFILE_BELOW_SCORE:
        return True, f"SCORE<{DISCOVERY_EVAL_SKIP_PROFILE_BELOW_SCORE:.2f}"
    return False, ""


def route_after_discovery_evaluate(
    state: PaperState,
) -> Literal["profile", "early_exit"]:
    """After scoring: enrich matrix unless early-exit thresholds say skip."""
    skip, _ = _discovery_skip_source_profile_after_eval(state)
    return "early_exit" if skip else "profile"


def discovery_eval_early_exit_node(state: PaperState) -> PaperState:
    """Trace + empty profile when scoring thresholds skip matrix extraction."""
    _, detail = _discovery_skip_source_profile_after_eval(state)
    return append_trace(
        {**state, "candidate_source_profile": {}},
        "discovery_eval_early_exit",
        "Early exit after scoring — skipped source profile.",
        detail=detail or "threshold",
        duration_ms=0.0,
        result={"reason": detail or "threshold"},
    )


def _discovery_profile_confidence_high(state: PaperState) -> bool:
    """High confidence → rule-based matrix; low confidence → LLM extraction."""
    if state.get("error"):
        return False
    if not state.get("candidate_fit"):
        return False
    score = float(state.get("candidate_score") or 0.0)
    return score >= DISCOVERY_PROFILE_RULE_SCORE_MIN


def _rule_based_discovery_source_profile(
    candidate: dict[str, Any],
    topic: str,
    *,
    eval_reason: str,
) -> dict[str, str]:
    """Populate evidence-matrix keys from OpenAlex metadata + evaluator reason only (no LLM)."""
    abstract = str(candidate.get("abstract") or "").strip()
    venue = str(candidate.get("venue") or "").strip()
    title = str(candidate.get("title") or "").strip()
    year = candidate.get("year")
    excerpt = abstract[:520] + ("…" if len(abstract) > 520 else "")
    yr = str(year) if year is not None else "N/A"
    reason_line = (eval_reason or "").strip() or title[:280]
    return {
        "authors": "N/A",
        "date_of_research": yr,
        "country_of_origin": "N/A",
        "purpose_aims": topic or "N/A",
        "research_questions": topic or "N/A",
        "data_used_method_collection_sample_size": "N/A",
        "methods_tools_used": "N/A",
        "method_and_data_collection_limitations": "N/A",
        "results": excerpt or "N/A",
        "contribution": reason_line[:600] or "N/A",
        "limitation_of_research_outcomes": "N/A",
        "future_perspectives": "N/A",
    }


def discovery_source_profile_node(state: PaperState) -> PaperState:
    """Structured evidence matrix: rule-based when confidence is high, else LLM."""
    if state.get("error"):
        return state
    candidate = state.get("current_candidate")
    if not candidate:
        return state
    topic = (state.get("topic") or "").strip()
    eval_reason = str(state.get("candidate_reason") or "")

    if _discovery_profile_confidence_high(state):
        source_profile = _rule_based_discovery_source_profile(
            candidate,
            topic,
            eval_reason=eval_reason,
        )
        return append_trace(
            {**state, "candidate_source_profile": source_profile},
            "discovery_source_profile",
            "Rule-based source profile (high confidence — no LLM).",
            detail=f"score≥{DISCOVERY_PROFILE_RULE_SCORE_MIN}, FIT=YES",
            duration_ms=0.0,
            result={
                "candidate_title": candidate.get("title", ""),
                "source_profile": source_profile,
                "profile_mode": "rule",
            },
        )

    try:
        prompt = DISCOVERY_SOURCE_PROFILE_PROMPT.format(
            topic=topic,
            title=candidate.get("title", ""),
            abstract=candidate.get("abstract", ""),
            venue=candidate.get("venue", ""),
            year=candidate.get("year", ""),
            cited_by_count=candidate.get("cited_by_count", 0),
        )
        t0 = time.perf_counter()
        content = invoke_gemini_prompt(prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        source_profile = _parse_source_profile(content)
        s2: PaperState = {
            **state,
            "candidate_source_profile": source_profile,
        }
        return append_trace(
            s2,
            "discovery_source_profile",
            "LLM source profile (low confidence — structured extraction).",
            duration_ms=elapsed_ms,
            result={
                "candidate_title": candidate.get("title", ""),
                "source_profile": source_profile,
                "profile_mode": "llm",
            },
        )
    except Exception as e:
        return append_trace(
            {**state, "error": f"Discovery source profile extraction failed: {str(e)}"},
            "discovery_source_profile",
            "Source profile extraction failed.",
            detail=str(e),
        )


def discovery_finalize_candidate_node(state: PaperState) -> PaperState:
    """Aggregate current candidate result into evaluated/qualified lists."""
    if state.get("error"):
        return state
    candidate = state.get("current_candidate")
    if not candidate:
        return state

    evaluated = list(state.get("evaluated_candidates") or [])
    selected = list(state.get("qualified_works") or [])
    score = float(state.get("candidate_score") or 0.0)
    fit = bool(state.get("candidate_fit"))
    quality = bool(state.get("candidate_quality"))
    reason = str(state.get("candidate_reason") or "")
    source_profile = dict(state.get("candidate_source_profile") or {})
    row = {
        **candidate,
        "score": score,
        "fit": fit,
        "quality": quality,
        "reason": reason,
        "source_profile": source_profile,
        "eval_duration_ms": state.get("candidate_eval_duration_ms"),
    }
    evaluated.append(row)
    if fit and quality and score >= 0.7:
        selected.append(row)
    s2: PaperState = {
        **state,
        "current_candidate": None,
        "candidate_score": 0.0,
        "candidate_fit": False,
        "candidate_quality": False,
        "candidate_reason": "",
        "candidate_source_profile": {},
        "candidate_eval_duration_ms": None,
        "evaluated_candidates": evaluated,
        "qualified_works": selected,
    }
    return append_trace(
        s2,
        "discovery_finalize_candidate",
        "Merged candidate evaluation into orchestrator state.",
        detail=f"qualified={len(selected)} / target={state.get('target_qualified_count', 2)}",
        result={
            "candidate": row,
            "qualified_count": len(selected),
            "evaluated_count": len(evaluated),
            "target": state.get("target_qualified_count", 2),
        },
    )


def route_discovery_candidate(state: PaperState) -> Literal["evaluate", "round_check", "end"]:
    """Route candidate loop: evaluate next candidate or move to round decision."""
    if state.get("error"):
        return "end"
    qualified = len(state.get("qualified_works") or [])
    target = int(state.get("target_qualified_count") or 2)
    if qualified >= target:
        return "end"
    if state.get("current_candidate"):
        return "evaluate"
    if state.get("candidate_queue"):
        return "evaluate"
    return "round_check"


def route_after_discovery_finalize(state: PaperState) -> Literal["pick", "round_check", "end"]:
    """Route after finalizing one candidate."""
    if state.get("error"):
        return "end"
    qualified = len(state.get("qualified_works") or [])
    target = int(state.get("target_qualified_count") or 2)
    if qualified >= target:
        return "end"
    if state.get("candidate_queue"):
        return "pick"
    return "round_check"


def discovery_round_check_node(state: PaperState) -> PaperState:
    """Trace boundary between candidate loop and round loop."""
    return append_trace(
        state,
        "discovery_round_check",
        "Finished this round queue; deciding whether to search another round.",
        detail=f"round={state.get('discovery_round', 0)}",
        result={
            "round": int(state.get("discovery_round") or 0),
            "qualified_count": len(state.get("qualified_works") or []),
            "target": int(state.get("target_qualified_count") or 2),
        },
    )


def route_discovery_loop(state: PaperState) -> Literal["search", "end"]:
    """Round loop until target qualified works or max rounds reached."""
    if state.get("error"):
        return "end"
    qualified = len(state.get("qualified_works") or [])
    target = int(state.get("target_qualified_count") or 2)
    rounds = int(state.get("discovery_round") or 0)
    max_rounds = int(state.get("max_discovery_rounds") or 5)
    if qualified >= target:
        return "end"
    if rounds >= max_rounds:
        return "end"
    return "search"


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
    """Helper to extract a section between two markers."""
    start_idx = text.find(start_marker)
    if start_idx == -1:
        return ""
    start_idx += len(start_marker)

    if end_marker:
        end_idx = text.find(end_marker, start_idx)
        if end_idx == -1:
            return text[start_idx:]
        return text[start_idx:end_idx]
    return text[start_idx:]


def _parse_yes_no(content: str, key: str) -> bool:
    for line in content.split("\n"):
        if line.upper().startswith(key.upper()):
            return "YES" in line.upper()
    return False


def _build_candidate_triage_block(candidates: list[dict]) -> str:
    lines: list[str] = []
    for idx, candidate in enumerate(candidates):
        title = str(candidate.get("title") or "").strip()
        abstract = str(candidate.get("abstract") or "").strip()
        if not abstract:
            abstract = "N/A"
        lines.append(f"[{idx}] TITLE: {title}")
        lines.append(f"[{idx}] ABSTRACT: {abstract[:1200]}")
    return "\n".join(lines)


def _build_pdf_excerpt(pdf_text: str, max_chars: int = 10000) -> str:
    text = (pdf_text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _parse_discovery_rank(content: str) -> tuple[list[int], bool, str]:
    ranked_indices: list[int] = []
    refetch = False
    reason = "No reason provided."
    for raw_line in content.split("\n"):
        line = raw_line.strip()
        upper = line.upper()
        if upper.startswith("ORDER:"):
            value = line.split(":", 1)[1].strip()
            parsed: list[int] = []
            for part in value.split(","):
                token = part.strip()
                if not token:
                    continue
                try:
                    parsed.append(int(token))
                except ValueError:
                    continue
            ranked_indices = parsed
        elif upper.startswith("REFETCH:"):
            refetch = "YES" in upper
        elif upper.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip() or reason
    deduped: list[int] = []
    seen: set[int] = set()
    for idx in ranked_indices:
        if idx in seen:
            continue
        seen.add(idx)
        deduped.append(idx)
    return deduped, refetch, reason


def _parse_source_profile(content: str) -> dict[str, str]:
    canonical_keys = [
        "authors",
        "date_of_research",
        "country_of_origin",
        "purpose_aims",
        "research_questions",
        "data_used_method_collection_sample_size",
        "methods_tools_used",
        "method_and_data_collection_limitations",
        "results",
        "contribution",
        "limitation_of_research_outcomes",
        "future_perspectives",
    ]
    profile: dict[str, str] = {k: "N/A" for k in canonical_keys}

    # 1) Primary path: parse declared SOURCE_PROFILE_JSON payload.
    json_profile = _parse_source_profile_json_block(content)
    if json_profile:
        for key in canonical_keys:
            value = str(json_profile.get(key, "")).strip()
            profile[key] = value or "N/A"
        return profile

    # 2) Fallback path: tolerant line parser for key:value outputs.
    label_to_key = {
        "AUTHORS": "authors",
        "DATE_OF_RESEARCH": "date_of_research",
        "COUNTRY_OF_ORIGIN": "country_of_origin",
        "PURPOSE_AIMS": "purpose_aims",
        "RESEARCH_QUESTIONS": "research_questions",
        "DATA_USED_METHOD_COLLECTION_SAMPLE_SIZE": "data_used_method_collection_sample_size",
        "METHODS_TOOLS_USED": "methods_tools_used",
        "METHOD_AND_DATA_COLLECTION_LIMITATIONS": "method_and_data_collection_limitations",
        "RESULTS": "results",
        "CONTRIBUTION": "contribution",
        "LIMITATION_OF_RESEARCH_OUTCOMES": "limitation_of_research_outcomes",
        "FUTURE_PERSPECTIVES": "future_perspectives",
        "AUTHOR_S": "authors",
        "DATE_OF_RESEARCH": "date_of_research",
    }
    for raw_line in content.split("\n"):
        line = raw_line.strip()
        if ":" not in line:
            continue
        lhs, rhs = line.split(":", 1)
        normalized_label = _normalize_profile_label(lhs)
        mapped = label_to_key.get(normalized_label)
        if mapped:
            value = rhs.strip()
            profile[mapped] = value or "N/A"
    return profile


def _parse_source_profile_json_block(content: str) -> dict[str, str] | None:
    marker = "SOURCE_PROFILE_JSON:"
    idx = content.find(marker)
    if idx == -1:
        return None
    after = content[idx + len(marker):].strip()
    if not after:
        return None
    obj = _extract_first_json_object(after)
    if not isinstance(obj, dict):
        return None
    return {str(k): str(v) for k, v in obj.items()}


def _extract_first_json_object(text: str) -> dict | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start:i + 1]
                try:
                    loaded = json.loads(snippet)
                except json.JSONDecodeError:
                    return None
                return loaded if isinstance(loaded, dict) else None
    return None


def _normalize_profile_label(label: str) -> str:
    chars: list[str] = []
    for ch in label.upper():
        if "A" <= ch <= "Z" or "0" <= ch <= "9":
            chars.append(ch)
        else:
            chars.append("_")
    normalized = "".join(chars).strip("_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized
