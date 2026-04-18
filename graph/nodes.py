from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from graph.state import PaperState
from utils.prompts import SUMMARISE_PROMPT, EVALUATE_PROMPT
import streamlit as st

def get_llm():
    """Initialise Claude via LangChain."""
    return ChatAnthropic(
        model="claude-opus-4-5",
        anthropic_api_key=st.secrets["ANTHROPIC_API_KEY"],
        max_tokens=2000,
    )

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
        llm = get_llm()
        prompt = SUMMARISE_PROMPT.format(pdf_text=state["pdf_text"])
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content

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
        llm = get_llm()
        research_focus = st.session_state.get("research_focus", "General ML and AI research")

        prompt = EVALUATE_PROMPT.format(
            research_focus=research_focus,
            summary=state["summary"],
            key_findings=state["key_findings"],
            methodology=state["methodology"],
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content

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
