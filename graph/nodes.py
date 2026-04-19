from graph.state import PaperState
from utils.prompts import SUMMARISE_PROMPT, EVALUATE_PROMPT
from utils.gemini_llm import invoke_gemini_prompt
import streamlit as st

def extract_node(state: PaperState) -> PaperState:
    """Node 1: Text is already extracted before the graph runs — just validate."""
    if not state.get("pdf_text") or len(state["pdf_text"].strip()) < 100:
        return {**state, "error": "Insufficient text extracted from PDF."}
    return state

def summarise_node(state: PaperState) -> PaperState:
    """Node 2: Summarise the paper and extract key info."""
    if state.get("error"):
        return state

    try:
        prompt = SUMMARISE_PROMPT.format(pdf_text=state["pdf_text"])
        content = invoke_gemini_prompt(prompt)

        # Parse structured response
        summary = _extract_section(content, "SUMMARY:", "KEY_FINDINGS:")
        key_findings = _extract_section(content, "KEY_FINDINGS:", "METHODOLOGY:")
        methodology = _extract_section(content, "METHODOLOGY:", None)

        return {
            **state,
            "summary": summary.strip(),
            "key_findings": key_findings.strip(),
            "methodology": methodology.strip(),
        }
    except Exception as e:
        return {**state, "error": f"Summarisation failed: {str(e)}"}

def evaluate_node(state: PaperState) -> PaperState:
    """Node 3: Score relevance against the research focus."""
    if state.get("error"):
        return state

    try:
        research_focus = st.session_state.get("research_focus", "General ML and AI research")

        prompt = EVALUATE_PROMPT.format(
            research_focus=research_focus,
            summary=state["summary"],
            key_findings=state["key_findings"],
            methodology=state["methodology"],
        )
        content = invoke_gemini_prompt(prompt)

        # Parse score
        score = 0.0
        fit = False
        reason = "Could not parse evaluation."

        for line in content.split("\n"):
            if line.startswith("SCORE:"):
                try:
                    score = float(line.replace("SCORE:", "").strip())
                except ValueError:
                    pass
            elif line.startswith("FIT:"):
                fit = "YES" in line.upper()
            elif line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()

        return {
            **state,
            "relevance_score": score,
            "fit": fit,
            "relevance_reason": reason,
        }
    except Exception as e:
        return {**state, "error": f"Evaluation failed: {str(e)}"}

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
