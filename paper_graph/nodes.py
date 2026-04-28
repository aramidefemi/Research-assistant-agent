import hashlib
import json
import re
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
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\\-_/]*")


def _llm_enabled(state: PaperState) -> bool:
    return bool(state.get("llm_enabled", True))


def _with_llm_used(state: PaperState) -> PaperState:
    return {
        **state,
        "llm_used": True,
        "fallback_reason": state.get("fallback_reason", ""),
    }


def _with_fallback_meta(state: PaperState, reason: str) -> PaperState:
    prior = (state.get("fallback_reason") or "").strip()
    merged = reason if not prior else (prior if reason in prior else f"{prior}; {reason}")
    return {
        **state,
        "llm_used": bool(state.get("llm_used", False)),
        "fallback_reason": merged,
    }


def _top_keywords(text: str, limit: int = 10) -> list[str]:
    stop = {
        "the", "and", "for", "that", "with", "from", "this", "these", "those",
        "into", "about", "using", "used", "their", "they", "there", "than", "then",
        "are", "was", "were", "been", "have", "has", "had", "can", "could",
        "will", "would", "should", "may", "might", "our", "your", "you", "its",
        "not", "but", "all", "any", "one", "two", "three", "paper", "study", "research",
        "result", "results", "method", "methods", "model", "models", "analysis", "data",
    }
    counts: dict[str, int] = {}
    for token in _WORD_RE.findall((text or "").lower()):
        if len(token) < 4 or token in stop:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [k for k, _ in ranked[:limit]]


def _fallback_summary_sections(pdf_text: str) -> tuple[str, str, str]:
    text = (pdf_text or "").strip()
    if not text:
        return ("No text available.", "No key findings extracted.", "No methodology extracted.")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    summary = " ".join(lines[:4])[:900] if lines else text[:900]
    findings = _top_keywords(text, limit=10)
    methodology_hits = [
        k for k in findings
        if any(x in k for x in ("method", "model", "regress", "network", "bayes", "boost", "transformer"))
    ]
    key_findings = ", ".join(findings) if findings else "No clear findings extracted."
    methodology = ", ".join(methodology_hits[:6]) if methodology_hits else "Methodology not explicitly detected."
    return summary, key_findings, methodology


def _deterministic_reason(research_focus: str, score: float, fit: bool) -> str:
    decision = "matches" if fit else "does not match"
    return (
        f"Deterministic fallback {decision} the research focus "
        f"(score={score:.2f}) based on keyword overlap with: {research_focus[:180]}"
    )


def _deterministic_source_profile_from_text(pdf_text: str, topic_hint: str = "") -> dict[str, str]:
    excerpt = _build_pdf_excerpt(pdf_text, max_chars=900)
    return {
        "authors": "N/A",
        "date_of_research": "N/A",
        "country_of_origin": "N/A",
        "purpose_aims": topic_hint or "N/A",
        "research_questions": topic_hint or "N/A",
        "data_used_method_collection_sample_size": "N/A",
        "methods_tools_used": "N/A",
        "method_and_data_collection_limitations": "N/A",
        "results": excerpt or "N/A",
        "contribution": "Deterministic fallback extraction (no LLM).",
        "limitation_of_research_outcomes": "N/A",
        "future_perspectives": "N/A",
    }


def _deterministic_score_fit_for_candidate(topic: str, candidate: dict[str, Any]) -> tuple[float, bool]:
    terms_topic = set(_top_keywords(topic, limit=14))
    if not terms_topic:
        return 0.0, False
    text = " ".join([
        str(candidate.get("title") or ""),
        str(candidate.get("abstract") or ""),
        str(candidate.get("venue") or ""),
    ])
    terms_paper = set(_top_keywords(text, limit=20))
    overlap = len(terms_topic.intersection(terms_paper))
    score = min(1.0, overlap / max(1, min(8, len(terms_topic))))
    return score, score >= 0.5


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

    if not _llm_enabled(state):
        summary, key_findings, methodology = _fallback_summary_sections(state["pdf_text"])
        s2 = _with_fallback_meta(
            {
                **state,
                "summary": summary,
                "key_findings": key_findings,
                "methodology": methodology,
            },
            "llm_disabled",
        )
        return append_trace(
            s2,
            "summarise",
            "Deterministic summarisation fallback (LLM disabled).",
            detail="No-LLM mode enabled by user toggle.",
            duration_ms=0.0,
            result={
                "summary": summary,
                "key_findings": key_findings,
                "methodology": methodology,
                "llm_used": False,
                "fallback_reason": s2.get("fallback_reason", ""),
            },
        )
    try:
        prompt = SUMMARISE_PROMPT.format(pdf_text=state["pdf_text"])
        t0 = time.perf_counter()
        content = invoke_gemini_prompt(prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Parse structured response
        summary = _extract_section(content, "SUMMARY:", "KEY_FINDINGS:")
        key_findings = _extract_section(content, "KEY_FINDINGS:", "METHODOLOGY:")
        methodology = _extract_section(content, "METHODOLOGY:", None)

        s2 = _with_llm_used({
            **state,
            "summary": summary.strip(),
            "key_findings": key_findings.strip(),
            "methodology": methodology.strip(),
        })
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
                "llm_used": True,
                "fallback_reason": s2.get("fallback_reason", ""),
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

    research_focus = st.session_state.get("research_focus", "General ML and AI research")
    if not _llm_enabled(state):
        score, fit = _deterministic_score_fit_for_candidate(
            str(research_focus),
            {
                "title": "",
                "abstract": " ".join(
                    [
                        str(state.get("summary") or ""),
                        str(state.get("key_findings") or ""),
                        str(state.get("methodology") or ""),
                    ]
                ),
            },
        )
        reason = (
            "Quick evaluation: narrative reasoning skipped."
            if st.session_state.get("evaluation_depth", "full") == "quick"
            else ""
        )
        ev: PaperState = _with_fallback_meta(
            {
                **state,
                "relevance_score": score,
                "fit": fit,
                "relevance_reason": reason,
            },
            "llm_disabled: score/fit fallback used",
        )
        fit_lbl = "YES" if fit else "NO"
        return append_trace(
            ev,
            "evaluate_score_fit",
            f"Deterministic relevance scoring fallback: SCORE={score:.2f}, FIT={fit_lbl}.",
            detail="No-LLM mode enabled.",
            duration_ms=0.0,
            result={
                "score": score,
                "fit": fit,
                "mode": "quick" if st.session_state.get("evaluation_depth", "full") == "quick" else "full",
                "llm_used": False,
                "fallback_reason": ev.get("fallback_reason", ""),
            },
        )
    try:
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
        ev: PaperState = _with_llm_used({
            **state,
            "relevance_score": score,
            "fit": fit,
            "relevance_reason": reason,
        })
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
                "llm_used": True,
                "fallback_reason": ev.get("fallback_reason", ""),
            },
        )
    except Exception as e:
        score, fit = _deterministic_score_fit_for_candidate(
            str(research_focus),
            {
                "title": "",
                "abstract": " ".join(
                    [
                        str(state.get("summary") or ""),
                        str(state.get("key_findings") or ""),
                        str(state.get("methodology") or ""),
                    ]
                ),
            },
        )
        ev = _with_fallback_meta(
            {
                **state,
                "relevance_score": score,
                "fit": fit,
            },
            f"llm_error: {str(e)}",
        )
        return append_trace(
            ev,
            "evaluate_score_fit",
            "Score/fit LLM failed; deterministic scoring fallback used.",
            detail=str(e)[:220],
            duration_ms=0.0,
            result={
                "score": score,
                "fit": fit,
                "mode": "quick" if st.session_state.get("evaluation_depth", "full") == "quick" else "full",
                "llm_used": bool(ev.get("llm_used", False)),
                "fallback_reason": ev.get("fallback_reason", ""),
            },
        )


def evaluate_reason_node(state: PaperState) -> PaperState:
    """Narrative justification (second evaluation step; optional via routing)."""
    if state.get("error"):
        return state

    if not _llm_enabled(state):
        reason = _deterministic_reason(
            st.session_state.get("research_focus", "General ML and AI research"),
            float(state.get("relevance_score") or 0.0),
            bool(state.get("fit")),
        )
        s2 = {**state, "relevance_reason": reason}
        s2 = _with_fallback_meta(s2, "llm_disabled: reason fallback used")
        return append_trace(
            s2,
            "evaluate_reason",
            "Deterministic reason fallback (LLM disabled).",
            detail="No-LLM mode enabled.",
            duration_ms=0.0,
            result={"reason": reason, "llm_used": False, "fallback_reason": s2.get("fallback_reason", "")},
        )
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
        ev: PaperState = _with_llm_used({**state, "relevance_reason": reason})
        excerpt = reason[:280] + ("…" if len(reason) > 280 else "")
        return append_trace(
            ev,
            "evaluate_reason",
            "Narrative explanation for the score and fit.",
            detail=f"{excerpt} | model call {elapsed_ms:.0f} ms.",
            duration_ms=elapsed_ms,
            result={"reason": reason, "llm_used": True, "fallback_reason": ev.get("fallback_reason", "")},
        )
    except Exception as e:
        err_state = _with_fallback_meta(state, f"llm_error: {str(e)}")
        reason = _deterministic_reason(
            st.session_state.get("research_focus", "General ML and AI research"),
            float(state.get("relevance_score") or 0.0),
            bool(state.get("fit")),
        )
        s2 = {**err_state, "relevance_reason": reason}
        return append_trace(
            s2,
            "evaluate_reason",
            "Reasoning LLM failed; deterministic reason fallback used.",
            detail=str(e)[:220],
            duration_ms=0.0,
            result={"reason": reason, "llm_used": bool(s2.get("llm_used", False)), "fallback_reason": s2.get("fallback_reason", "")},
        )


def evaluate_matrix_node(state: PaperState) -> PaperState:
    """Structured evidence matrix extraction (dedicated evaluation step)."""
    if state.get("error"):
        return state

    if not _llm_enabled(state):
        source_profile = _deterministic_source_profile_from_text(
            state.get("pdf_text", ""),
            topic_hint=str(st.session_state.get("research_focus", "")).strip(),
        )
        s2: PaperState = _with_fallback_meta({**state, "source_profile": source_profile}, "llm_disabled: matrix fallback used")
        return append_trace(
            s2,
            "evaluate_matrix",
            "Deterministic source profile fallback (LLM disabled).",
            detail="No-LLM mode enabled.",
            duration_ms=0.0,
            result={"source_profile": source_profile, "llm_used": False, "fallback_reason": s2.get("fallback_reason", "")},
        )

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
        s2: PaperState = _with_llm_used({**state, "source_profile": source_profile})
        return append_trace(
            s2,
            "evaluate_matrix",
            "Extracted dedicated evaluation matrix fields from paper content.",
            duration_ms=elapsed_ms,
            result={"source_profile": source_profile, "llm_used": True, "fallback_reason": s2.get("fallback_reason", "")},
        )
    except Exception as e:
        err_state = _with_fallback_meta(state, f"llm_error: {str(e)}")
        source_profile = _deterministic_source_profile_from_text(
            state.get("pdf_text", ""),
            topic_hint=str(st.session_state.get("research_focus", "")).strip(),
        )
        return append_trace(
            {**err_state, "source_profile": source_profile},
            "evaluate_matrix",
            "Matrix LLM failed; deterministic source profile fallback used.",
            detail=str(e),
            duration_ms=0.0,
            result={"source_profile": source_profile, "llm_used": bool(err_state.get("llm_used", False)), "fallback_reason": err_state.get("fallback_reason", "")},
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
    if not _llm_enabled(state):
        s2 = _with_fallback_meta(state, "llm_disabled: discovery triage fallback to search order")
        return append_trace(
            s2,
            "discovery_triage_candidates",
            "No-LLM mode: skipped abstract triage and kept search order.",
            detail="Deterministic fallback uses source search ordering.",
            duration_ms=0.0,
            result={
                "ranked_count": len(candidates),
                "refetch": False,
                "fallback": "search_order",
                "llm_used": False,
                "fallback_reason": s2.get("fallback_reason", ""),
            },
        )
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
    if not _llm_enabled(state):
        score = _score_focus_overlap(
            f"{candidate.get('title', '')} {candidate.get('abstract', '')}",
            topic,
        )
        fit = score >= 0.25
        quality = bool(candidate.get("year") and int(candidate.get("year")) >= 2010)
        reason = (
            f"Deterministic fallback: lexical topic overlap={score:.2f}; "
            f"quality={'YES' if quality else 'NO'} from metadata heuristics."
        )
        s2 = _with_fallback_meta(
            {
                **state,
                "candidate_score": score,
                "candidate_fit": fit,
                "candidate_quality": quality,
                "candidate_reason": reason,
                "candidate_eval_duration_ms": 0.0,
            },
            "llm_disabled: candidate evaluation fallback used",
        )
        return append_trace(
            s2,
            "discovery_evaluate_candidate",
            f"Deterministic candidate evaluation: SCORE={score:.2f}, FIT={'YES' if fit else 'NO'}, "
            f"QUALITY={'YES' if quality else 'NO'}.",
            detail=reason[:200],
            duration_ms=0.0,
            result={
                "candidate_title": candidate.get("title", ""),
                "score": score,
                "fit": fit,
                "quality": quality,
                "reason": reason,
                "cache_hit": False,
                "llm_used": False,
                "fallback_reason": s2.get("fallback_reason", ""),
            },
        )

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


def _extract_methodology_risk_flags(
    text: str,
    *,
    context_label: str,
) -> list[dict[str, str]]:
    low = (text or "").lower()
    if not low:
        return []
    rules: list[tuple[str, str, str]] = [
        ("small_sample_size", "Small sample size", "small sample"),
        ("weak_baseline", "Weak or missing baseline", "baseline"),
        ("missing_ablations", "Missing or limited ablations", "ablation"),
        ("unclear_evaluation", "Unclear evaluation setup", "evaluation"),
    ]
    flags: list[dict[str, str]] = []
    for code, label, needle in rules:
        idx = low.find(needle)
        if idx == -1:
            continue
        start = max(0, idx - 80)
        end = min(len(text), idx + len(needle) + 120)
        snippet = (text[start:end] or "").strip().replace("\n", " ")
        if len(snippet) > 260:
            snippet = snippet[:257] + "..."
        flags.append(
            {
                "code": code,
                "label": label,
                "evidence": snippet or f"Matched keyword: {needle}",
                "source": context_label,
            }
        )
    return flags


def _build_citation_use_examples(
    *,
    title: str,
    summary: str,
    key_findings: str,
    methodology: str,
    source_profile: dict[str, str],
    risk_flags: list[dict[str, str]],
) -> list[str]:
    method = (
        (source_profile.get("methods_tools_used") or "").strip()
        or methodology.strip()
        or "the paper's described approach"
    )
    results = (
        (source_profile.get("results") or "").strip()
        or key_findings.strip()
        or summary.strip()
        or "reported outcomes"
    )
    limitation = (
        (source_profile.get("limitation_of_research_outcomes") or "").strip()
        or (risk_flags[0].get("evidence", "").strip() if risk_flags else "")
        or "the paper does not provide a clear limitation statement"
    )
    paper_name = title.strip() or "this paper"
    return [
        f"Method reuse: adapt {method[:220]} from {paper_name} as a baseline implementation in your workflow.",
        f"Benchmarking: compare your metrics directly against {results[:240]} from {paper_name}, matching setup as closely as possible.",
        f"Research-gap extension: design a follow-up experiment addressing this limitation from {paper_name}: {limitation[:240]}.",
    ]


def _build_pdf_citation_use_examples(state: PaperState) -> list[str]:
    return _build_citation_use_examples(
        title=str(state.get("filename") or ""),
        summary=str(state.get("summary") or ""),
        key_findings=str(state.get("key_findings") or ""),
        methodology=str(state.get("methodology") or ""),
        source_profile=dict(state.get("source_profile") or {}),
        risk_flags=list(state.get("risk_flags") or []),
    )


def _confidence_label_from_signals(score: float, risk_count: int) -> str:
    if score >= 0.75 and risk_count <= 1:
        return "high"
    if score >= 0.45 and risk_count <= 3:
        return "medium"
    return "low"


def _build_claim_evidence_map_for_pdf(
    *,
    score: float,
    fit: bool,
    reason: str,
    summary: str,
    key_findings: str,
    methodology: str,
) -> list[dict[str, str]]:
    score_claim = f"Paper fit={ 'YES' if fit else 'NO' } with relevance score={score:.2f}."
    return [
        {
            "claim": score_claim,
            "evidence": (reason or "No explicit reason returned.")[:260],
            "source": "evaluation_reason",
        },
        {
            "claim": "Summary captures main contribution.",
            "evidence": (summary or "N/A")[:260],
            "source": "summary",
        },
        {
            "claim": "Key findings were extracted from the paper.",
            "evidence": (key_findings or "N/A")[:260],
            "source": "key_findings",
        },
        {
            "claim": "Methodological detail supports interpretation.",
            "evidence": (methodology or "N/A")[:260],
            "source": "methodology",
        },
    ]


def _build_claim_evidence_map_for_discovery(
    *,
    score: float,
    fit: bool,
    quality: bool,
    reason: str,
    abstract: str,
    source_profile: dict[str, str],
) -> list[dict[str, str]]:
    return [
        {
            "claim": f"Candidate fit={ 'YES' if fit else 'NO' } quality={ 'YES' if quality else 'NO' } score={score:.2f}.",
            "evidence": (reason or "No explicit reason returned.")[:260],
            "source": "candidate_reason",
        },
        {
            "claim": "Abstract supports topical relevance assessment.",
            "evidence": (abstract or "N/A")[:260],
            "source": "abstract",
        },
        {
            "claim": "Reported results support qualification decision.",
            "evidence": str((source_profile.get("results") or "N/A"))[:260],
            "source": "source_profile.results",
        },
    ]


def _build_pdf_evidence_contract(state: PaperState, risk_flags: list[dict[str, str]]) -> dict[str, Any]:
    score = float(state.get("relevance_score") or 0.0)
    claims = _build_claim_evidence_map_for_pdf(
        score=score,
        fit=bool(state.get("fit")),
        reason=str(state.get("relevance_reason") or ""),
        summary=str(state.get("summary") or ""),
        key_findings=str(state.get("key_findings") or ""),
        methodology=str(state.get("methodology") or ""),
    )
    return {
        "confidence_label": _confidence_label_from_signals(score, len(risk_flags)),
        "insufficient_evidence": not bool(claims),
        "claim_evidence": claims,
    }


def _build_discovery_evidence_contract(
    *,
    score: float,
    fit: bool,
    quality: bool,
    reason: str,
    abstract: str,
    source_profile: dict[str, str],
    risk_flags: list[dict[str, str]],
) -> dict[str, Any]:
    claims = _build_claim_evidence_map_for_discovery(
        score=score,
        fit=fit,
        quality=quality,
        reason=reason,
        abstract=abstract,
        source_profile=source_profile,
    )
    return {
        "confidence_label": _confidence_label_from_signals(score, len(risk_flags)),
        "insufficient_evidence": not bool(claims),
        "claim_evidence": claims,
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

    if not _llm_enabled(state):
        source_profile = _rule_based_discovery_source_profile(
            candidate,
            topic,
            eval_reason=eval_reason,
        )
        s2 = _with_fallback_meta(
            {**state, "candidate_source_profile": source_profile},
            "llm_disabled: discovery source profile fallback used",
        )
        return append_trace(
            s2,
            "discovery_source_profile",
            "No-LLM mode: deterministic source profile from candidate metadata.",
            detail="Derived from title/abstract/venue/year fields.",
            duration_ms=0.0,
            result={
                "candidate_title": candidate.get("title", ""),
                "source_profile": source_profile,
                "profile_mode": "rule",
                "llm_used": False,
                "fallback_reason": s2.get("fallback_reason", ""),
            },
        )

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

    if not _llm_enabled(state):
        source_profile = _rule_based_discovery_source_profile(
            candidate,
            topic,
            eval_reason=eval_reason,
        )
        s2 = _with_fallback_meta({**state, "candidate_source_profile": source_profile}, "llm_disabled: discovery profile fallback used")
        return append_trace(
            s2,
            "discovery_source_profile",
            "Deterministic discovery source profile fallback (LLM disabled).",
            detail="No-LLM mode enabled.",
            duration_ms=0.0,
            result={
                "candidate_title": candidate.get("title", ""),
                "source_profile": source_profile,
                "profile_mode": "rule_no_llm",
                "llm_used": False,
                "fallback_reason": s2.get("fallback_reason", ""),
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
        s2: PaperState = _with_llm_used({**state, "candidate_source_profile": source_profile})
        return append_trace(
            s2,
            "discovery_source_profile",
            "LLM source profile (low confidence — structured extraction).",
            duration_ms=elapsed_ms,
            result={
                "candidate_title": candidate.get("title", ""),
                "source_profile": source_profile,
                "profile_mode": "llm",
                "llm_used": True,
                "fallback_reason": s2.get("fallback_reason", ""),
            },
        )
    except Exception as e:
        source_profile = _rule_based_discovery_source_profile(
            candidate,
            topic,
            eval_reason=eval_reason,
        )
        err_state = _with_fallback_meta({**state, "candidate_source_profile": source_profile}, f"llm_error: {str(e)}")
        return append_trace(
            err_state,
            "discovery_source_profile",
            "Source profile LLM failed; deterministic fallback used.",
            detail=str(e)[:220],
            duration_ms=0.0,
            result={
                "candidate_title": candidate.get("title", ""),
                "source_profile": source_profile,
                "profile_mode": "rule_error_fallback",
                "llm_used": bool(err_state.get("llm_used", False)),
                "fallback_reason": err_state.get("fallback_reason", ""),
            },
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
    risk_flags = _extract_methodology_risk_flags(
        title=str(candidate.get("title") or ""),
        abstract=str(candidate.get("abstract") or ""),
        source_profile=source_profile,
        methodology_text=str(source_profile.get("methods_tools_used") or ""),
    )
    citation_use_examples = _build_citation_use_examples(
        title=str(candidate.get("title") or ""),
        summary=str(candidate.get("abstract") or ""),
        key_findings=reason,
        methodology=str(source_profile.get("methods_tools_used") or ""),
        source_profile=source_profile,
        risk_flags=risk_flags,
    )
    evidence_contract = _build_discovery_evidence_contract(
        score=score,
        fit=fit,
        quality=quality,
        reason=reason,
        abstract=str(candidate.get("abstract") or ""),
        source_profile=source_profile,
        risk_flags=risk_flags,
    )
    row = {
        **candidate,
        "score": score,
        "fit": fit,
        "quality": quality,
        "reason": reason,
        "source_profile": source_profile,
        "risk_flags": risk_flags,
        "citation_use_examples": citation_use_examples,
        "evidence_contract": evidence_contract,
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
            "citation_use_examples": citation_use_examples,
            "evidence_contract": evidence_contract,
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
