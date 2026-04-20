import time

import streamlit as st

from graph.state import PaperState
from graph.trace import append_trace
from typing import Literal

from utils.prompts import (
    SUMMARISE_PROMPT,
    EVALUATE_SCORE_FIT_PROMPT,
    EVALUATE_REASON_PROMPT,
    DISCOVERY_SCORE_FIT_PROMPT,
    DISCOVERY_QUALITY_REASON_PROMPT,
)
from utils.gemini_llm import invoke_gemini_prompt
from utils.journal_search import search_journals


def extract_node(state: PaperState) -> PaperState:
    """Node 1: Text is already extracted before the graph runs — just validate."""
    text = (state.get("pdf_text") or "").strip()
    if len(text) < 100:
        return append_trace(
            {**state, "error": "Insufficient text extracted from PDF."},
            "extract",
            "Stopped pipeline — unusable PDF text.",
            detail="Fewer than 100 non-whitespace characters.",
        )
    return append_trace(
        state,
        "extract",
        "Validated corpus; handed off to summarisation.",
        detail=f"{len(text):,} characters of extracted text.",
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
        )
    except Exception as e:
        return append_trace(
            {**state, "error": f"Evaluation reasoning failed: {str(e)}"},
            "evaluate_reason",
            "Reasoning step failed.",
            detail=str(e),
        )


def route_evaluation_after_score_fit(state: PaperState) -> Literal["reason", "end"]:
    """Orchestration: full evaluation runs a second LLM for REASON; quick stops after score/fit."""
    if state.get("error"):
        return "end"
    if st.session_state.get("evaluation_depth", "full") == "quick":
        return "end"
    return "reason"


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
    )


def discovery_pick_candidate_node(state: PaperState) -> PaperState:
    """Pick next candidate from queue."""
    if state.get("error"):
        return state
    queue = list(state.get("candidate_queue") or [])
    current = queue.pop(0) if queue else None
    s2: PaperState = {
        **state,
        "candidate_queue": queue,
        "current_candidate": current,
    }
    if not current:
        return append_trace(
            s2,
            "discovery_pick_candidate",
            "No more candidates in current round queue.",
        )
    return append_trace(
        s2,
        "discovery_pick_candidate",
        "Picked one candidate for evaluation.",
        detail=current.get("title", "")[:120],
    )


def discovery_score_fit_node(state: PaperState) -> PaperState:
    """Evaluate score + fit for current candidate."""
    if state.get("error"):
        return state

    topic = (state.get("topic") or "").strip()
    candidate = state.get("current_candidate")
    if not candidate:
        return state

    try:
        prompt = DISCOVERY_SCORE_FIT_PROMPT.format(
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
        s2: PaperState = {
            **state,
            "candidate_score": score,
            "candidate_fit": fit,
        }
        return append_trace(
            s2,
            "discovery_score_fit",
            f"Scored current candidate: SCORE={score:.2f}, FIT={'YES' if fit else 'NO'}.",
            duration_ms=elapsed_ms,
        )
    except Exception as e:
        return append_trace(
            {**state, "error": f"Discovery score/fit failed: {str(e)}"},
            "discovery_score_fit",
            "Candidate score/fit evaluation failed.",
            detail=str(e),
        )


def discovery_quality_reason_node(state: PaperState) -> PaperState:
    """Evaluate quality + reason for current candidate."""
    if state.get("error"):
        return state

    topic = (state.get("topic") or "").strip()
    candidate = state.get("current_candidate")
    if not candidate:
        return state

    try:
        score = float(state.get("candidate_score") or 0.0)
        fit = bool(state.get("candidate_fit"))
        prompt = DISCOVERY_QUALITY_REASON_PROMPT.format(
            topic=topic,
            title=candidate.get("title", ""),
            abstract=candidate.get("abstract", ""),
            venue=candidate.get("venue", ""),
            year=candidate.get("year", ""),
            cited_by_count=candidate.get("cited_by_count", 0),
            score=score,
            fit_label="YES" if fit else "NO",
        )
        t0 = time.perf_counter()
        content = invoke_gemini_prompt(prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        quality = _parse_yes_no(content, "QUALITY:")
        reason = _parse_reason(content)
        s2: PaperState = {
            **state,
            "candidate_quality": quality,
            "candidate_reason": reason,
            "candidate_eval_duration_ms": elapsed_ms,
        }
        return append_trace(
            s2,
            "discovery_quality_reason",
            f"Assessed scholarly quality: QUALITY={'YES' if quality else 'NO'}.",
            detail=reason[:200],
            duration_ms=elapsed_ms,
        )
    except Exception as e:
        return append_trace(
            {**state, "error": f"Discovery quality/reason failed: {str(e)}"},
            "discovery_quality_reason",
            "Candidate quality/reason evaluation failed.",
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
    row = {
        **candidate,
        "score": score,
        "fit": fit,
        "quality": quality,
        "reason": reason,
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
        "candidate_eval_duration_ms": None,
        "evaluated_candidates": evaluated,
        "qualified_works": selected,
    }
    return append_trace(
        s2,
        "discovery_finalize_candidate",
        "Merged candidate evaluation into orchestrator state.",
        detail=f"qualified={len(selected)} / target={state.get('target_qualified_count', 2)}",
    )


def route_discovery_candidate(state: PaperState) -> Literal["score_fit", "round_check", "end"]:
    """Route candidate loop: evaluate next candidate or move to round decision."""
    if state.get("error"):
        return "end"
    qualified = len(state.get("qualified_works") or [])
    target = int(state.get("target_qualified_count") or 2)
    if qualified >= target:
        return "end"
    if state.get("current_candidate"):
        return "score_fit"
    if state.get("candidate_queue"):
        return "score_fit"
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
