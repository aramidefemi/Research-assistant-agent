import time

import streamlit as st

from graph.state import PaperState
from graph.trace import append_trace
from typing import Literal

from utils.prompts import SUMMARISE_PROMPT, EVALUATE_SCORE_FIT_PROMPT, EVALUATE_REASON_PROMPT
from utils.gemini_llm import invoke_gemini_prompt


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
