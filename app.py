try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv() -> bool:
        return False

load_dotenv()

import streamlit as st
from html import escape
from typing import Any
from collections.abc import Iterator
from utils.pdf_reader import extract_text_from_pdf
from utils.trace_store import persist_pipeline_run, log_app_error
try:
    from utils.trace_store import log_usage_event
except Exception:
    def log_usage_event(event_type: str, payload: dict[str, Any] | None = None) -> None:
        return None
try:
    from utils.nav import render_page_nav
except Exception:
    def render_page_nav() -> None:
        st.markdown("[Research workspace](/) · [Admin usage stats](/Admin_Usage_Stats)")
        return None
from utils.trace_flowchart import build_trace_flowchart_dot
from utils.gemini_llm import invoke_gemini_prompt, invoke_gemini_prompt_stream
from paper_graph.pipeline import pipeline, discovery_pipeline
from paper_graph.trace import trace_step_title

def _mat_html(icon: str) -> str:
    return f'<span class="material-sym" aria-hidden="true">{icon}</span>'


def _html(value: Any) -> str:
    return escape("" if value is None else str(value))


def _clamp_score(score: float) -> int:
    return max(0, min(100, int(round(score * 100))))


def _build_section_intro_html(eyebrow: str, title: str, copy: str) -> str:
    return (
        f'<div class="ra-section-intro">'
        f'<div class="ra-eyebrow">{_html(eyebrow)}</div>'
        f'<h2 class="ra-section-title">{_html(title)}</h2>'
        f'<p class="ra-section-copy">{_html(copy)}</p>'
        f"</div>"
    )


def _build_stat_card_html(label: str, value: str, hint: str | None = None) -> str:
    hint_html = f'<div class="ra-stat-hint">{_html(hint)}</div>' if hint else ""
    return (
        f'<div class="ra-stat-card">'
        f'<div class="ra-stat-label">{_html(label)}</div>'
        f'<div class="ra-stat-value">{_html(value)}</div>'
        f"{hint_html}"
        f"</div>"
    )


def _build_stats_grid_html(stats: list[tuple[str, str, str | None]]) -> str:
    return '<div class="ra-stat-grid">' + "".join(
        _build_stat_card_html(label, value, hint) for label, value, hint in stats
    ) + "</div>"


def _build_info_panel_html(title: str, copy: str, items: list[str]) -> str:
    items_html = "".join(f"<li>{_html(item)}</li>" for item in items)
    return (
        f'<div class="ra-info-panel">'
        f'<div class="ra-eyebrow">Workspace guide</div>'
        f'<h3 class="ra-panel-title">{_html(title)}</h3>'
        f'<p class="ra-panel-copy">{_html(copy)}</p>'
        f'<ul class="ra-info-list">{items_html}</ul>'
        f"</div>"
    )


def _build_badge_html(label: str, tone: str = "neutral", icon: str | None = None) -> str:
    icon_html = _mat_html(icon) if icon else ""
    return f'<span class="ra-badge ra-badge--{tone}">{icon_html}{_html(label)}</span>'


def _build_meta_row_html(items: list[tuple[str, str]]) -> str:
    chips = "".join(
        f'<div class="ra-meta-chip"><span>{_html(label)}</span><strong>{_html(value)}</strong></div>'
        for label, value in items
        if value
    )
    return f'<div class="ra-meta-row">{chips}</div>' if chips else ""


def _build_progress_html(score: float, tone: str) -> str:
    return (
        '<div class="ra-progress">'
        f'<div class="ra-progress-bar ra-progress-bar--{tone}" style="width:{_clamp_score(score)}%;"></div>'
        "</div>"
    )


def _build_summary_card_html(
    *,
    title: str,
    eyebrow: str,
    badges: list[str],
    score: float,
    score_label: str,
    reason_label: str,
    reason: str,
    metadata: list[tuple[str, str]],
    tone: str,
) -> str:
    badge_html = "".join(badges)
    meta_html = _build_meta_row_html(metadata)
    score_pct = _clamp_score(score)
    return (
        f'<div class="ra-summary-card">'
        f'<div class="ra-summary-head">'
        f'<div><div class="ra-eyebrow">{_html(eyebrow)}</div>'
        f'<h3 class="ra-summary-title">{_html(title)}</h3></div>'
        f'<div class="ra-badge-row">{badge_html}</div>'
        f"</div>"
        f'<div class="ra-summary-grid">'
        f'<div class="ra-score-card">'
        f'<div class="ra-score-label">{_html(score_label)}</div>'
        f'<div class="ra-score-value">{score_pct}<span>/100</span></div>'
        f"{_build_progress_html(score, tone)}"
        f"</div>"
        f'<div class="ra-reason-card">'
        f'<div class="ra-score-label">{_html(reason_label)}</div>'
        f'<p class="ra-reason-copy">{_html(reason or "No rationale returned.")}</p>'
        f"{meta_html}"
        f"</div>"
        f"</div>"
        f"</div>"
    )


def _render_risk_flags(flags: list[dict[str, str]]) -> None:
    if not flags:
        st.caption("No explicit risk flags found.")
        return
    for i, flag in enumerate(flags, start=1):
        label = str(flag.get("label") or "Risk flag")
        evidence = str(flag.get("evidence") or "No evidence snippet available.")
        st.markdown(f"**{i}. {label}**")
        st.caption(evidence)


def _render_citation_use_examples(examples: list[str]) -> None:
    cleaned = [str(x).strip() for x in examples if str(x).strip()]
    if not cleaned:
        st.caption("No citation-use examples generated.")
        return
    for idx, example in enumerate(cleaned, start=1):
        st.markdown(f"{idx}. {example}")


def _render_evidence_contract(contract: dict[str, Any]) -> None:
    if not contract:
        st.caption("No evidence contract available.")
        return
    confidence = str(contract.get("confidence_label") or "low")
    insufficient = bool(contract.get("insufficient_evidence"))
    st.markdown(f"**Confidence:** `{confidence}`")
    st.markdown(f"**Insufficient evidence:** `{'yes' if insufficient else 'no'}`")
    claims = list(contract.get("claim_evidence") or [])
    if not claims:
        st.caption("No claim-to-evidence mappings found.")
        return
    for i, row in enumerate(claims, start=1):
        claim = str(row.get("claim") or "Claim")
        evidence = str(row.get("evidence") or "No evidence snippet.")
        source = str(row.get("source") or "unknown")
        st.markdown(f"**{i}. {claim}**")
        st.caption(f"{evidence} [{source}]")


SOURCE_MATRIX_FIELDS = [
    ("authors", "Author/s"),
    ("date_of_research", "Date of research"),
    ("country_of_origin", "Country of origin"),
    ("purpose_aims", "Purpose/Aims"),
    ("research_questions", "Research questions"),
    ("data_used_method_collection_sample_size", "Data used / method of data collection / sample size"),
    ("methods_tools_used", "Methods/tools used"),
    ("method_and_data_collection_limitations", "Method and data collection limitations"),
    ("results", "Results"),
    ("contribution", "Contribution"),
    ("limitation_of_research_outcomes", "Limitation of research outcomes"),
    ("future_perspectives", "Future perspectives"),
]


def _format_duration_ms(ms: float) -> str:
    sec = max(0, round(ms / 1000.0))
    if sec < 60:
        return f"{sec}s"
    m, s = divmod(sec, 60)
    return f"{m}m {s}s" if s else f"{m}m"


def _render_stage_result(value: Any, key: str, *, depth: int = 0) -> None:
    if depth > 3:
        st.caption("Max nested depth reached.")
        return
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            label = f"{key}.{child_key}" if key else child_key
            if isinstance(child_value, (dict, list)):
                with st.expander(label, expanded=False):
                    _render_stage_result(child_value, label, depth=depth + 1)
            else:
                st.markdown(f"**{label}:** `{child_value}`")
        return
    if isinstance(value, list):
        if not value:
            st.caption(f"{key}: []")
            return
        for idx, item in enumerate(value, start=1):
            item_label = f"{key}[{idx}]"
            if isinstance(item, (dict, list)):
                with st.expander(item_label, expanded=False):
                    _render_stage_result(item, item_label, depth=depth + 1)
            else:
                st.markdown(f"**{item_label}:** `{item}`")
        return
    st.markdown(f"**{key}:** `{value}`")


def write_trace_steps(trace: list, *, idle_msg: str | None = "No trace entries for this run.") -> None:
    """Render agent trace steps into the current Streamlit container."""
    if not trace:
        if idle_msg:
            st.info(idle_msg)
        return
    total_ms = sum(s.get("duration_ms") or 0 for s in trace if s.get("duration_ms") is not None)
    if total_ms > 0:
        st.caption(f"LLM-timed steps total ≈ **{_format_duration_ms(total_ms)}**")
    for i, step in enumerate(trace, start=1):
        node = step.get("node", "?")
        title = trace_step_title(str(node))
        with st.expander(f"Step {i} — {title}", expanded=(i <= 2)):
            st.markdown(step.get("contribution", ""))
            det = step.get("detail") or ""
            if det:
                st.caption(det)
            dm = step.get("duration_ms")
            if dm is not None:
                st.caption(f":material/schedule: {_format_duration_ms(dm)}")
            result = step.get("result")
            if result:
                with st.expander("Stage result", expanded=False):
                    _render_stage_result(result, "")


def write_trace_flowchart(trace: list[dict[str, Any]]) -> None:
    """Render executed trace as a runtime flowchart."""
    st.graphviz_chart(build_trace_flowchart_dot(trace))


def _render_source_matrix(profile: dict[str, str]) -> None:
    rows: list[dict[str, str]] = []
    for key, label in SOURCE_MATRIX_FIELDS:
        rows.append(
            {
                "Column": label,
                "Extracted value": (profile.get(key) or "N/A"),
            }
        )
    st.table(rows)


def _split_sentences(text: str) -> list[str]:
    raw = (text or "").replace("\n", " ").strip()
    if not raw:
        return []
    parts: list[str] = []
    buf = []
    for ch in raw:
        buf.append(ch)
        if ch in ".!?":
            sentence = "".join(buf).strip()
            if sentence:
                parts.append(sentence)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _keyword_overlap_score(question: str, sentence: str) -> int:
    q_terms = {w.lower() for w in question.split() if len(w) >= 4}
    if not q_terms:
        return 0
    s_terms = {w.lower().strip(".,:;()[]{}") for w in sentence.split()}
    return len(q_terms.intersection(s_terms))


def _deterministic_chat_answer(question: str, docs: list[dict[str, str]]) -> str:
    ranked: list[tuple[int, str, str]] = []
    for doc in docs:
        source = doc["source"]
        for sent in _split_sentences(doc["text"]):
            score = _keyword_overlap_score(question, sent)
            if score > 0:
                ranked.append((score, sent, source))
    if not ranked:
        return (
            "Insufficient evidence in selected papers to answer confidently.\n\n"
            "Citations:\n- [none]"
        )
    ranked.sort(key=lambda x: x[0], reverse=True)
    top = ranked[:3]
    bullets = "\n".join(f"- {sent} [{src}]" for _, sent, src in top)
    cites = "\n".join(f"- [{src}]" for _, _, src in top)
    return f"Best evidence from selected papers:\n{bullets}\n\nCitations:\n{cites}"


def _build_chat_context_docs() -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    for result in st.session_state.results:
        if result.get("error"):
            continue
        text = " ".join(
            [
                str(result.get("summary") or ""),
                str(result.get("key_findings") or ""),
                str(result.get("methodology") or ""),
            ]
        ).strip()
        if not text:
            continue
        docs.append({"source": str(result.get("filename") or "pdf"), "text": text})
    for run in st.session_state.discovery_results:
        for item in (run.get("qualified_works") or []):
            text = " ".join(
                [
                    str(item.get("abstract") or ""),
                    str((item.get("source_profile") or {}).get("results") or ""),
                    str(item.get("reason") or ""),
                ]
            ).strip()
            if not text:
                continue
            docs.append({"source": str(item.get("title") or "discovery"), "text": text})
    return docs


def _chat_state_key(selected_sources: list[str], focus: str) -> str:
    src = "|".join(sorted(selected_sources))
    return f"chat::{focus.strip()}::{src}"


def _answer_paper_chat(
    question: str,
    docs: list[dict[str, str]],
    focus: str,
    llm_enabled: bool,
) -> tuple[str, bool, str]:
    if not llm_enabled:
        return (
            _deterministic_chat_answer(question, docs),
            False,
            "llm_disabled: chat fallback used",
        )
    context_block = "\n\n".join(
        f"SOURCE: {d['source']}\nTEXT:\n{d['text'][:2200]}" for d in docs
    )
    prompt = (
        "You are a research assistant. Answer only from provided sources.\n"
        "If evidence is missing, say: 'Insufficient evidence.'\n"
        "Include a final 'Citations:' section with [SOURCE] anchors.\n\n"
        f"Research focus:\n{focus or 'N/A'}\n\n"
        f"Question:\n{question}\n\n"
        f"Sources:\n{context_block}\n"
    )
    try:
        return (invoke_gemini_prompt(prompt), True, "")
    except Exception as e:
        return (
            _deterministic_chat_answer(question, docs),
            False,
            f"llm_error: {str(e)}",
        )


def _answer_paper_chat_stream(
    question: str,
    docs: list[dict[str, str]],
    focus: str,
    llm_enabled: bool,
) -> tuple[Iterator[str], bool, str]:
    if not llm_enabled:
        return (
            iter([_deterministic_chat_answer(question, docs)]),
            False,
            "llm_disabled: chat fallback used",
        )
    context_block = "\n\n".join(
        f"SOURCE: {d['source']}\nTEXT:\n{d['text'][:2200]}" for d in docs
    )
    prompt = (
        "You are a research assistant. Answer only from provided sources.\n"
        "If evidence is missing, say: 'Insufficient evidence.'\n"
        "Include a final 'Citations:' section with [SOURCE] anchors.\n\n"
        f"Research focus:\n{focus or 'N/A'}\n\n"
        f"Question:\n{question}\n\n"
        f"Sources:\n{context_block}\n"
    )
    try:
        return (invoke_gemini_prompt_stream(prompt), True, "")
    except Exception as e:
        return (
            iter([_deterministic_chat_answer(question, docs)]),
            False,
            f"llm_error: {str(e)}",
        )


def run_pipeline_stream(graph, initial_state: dict, live_placeholder, run_label: str) -> dict:
    """Run a graph with streaming; updates `live_placeholder` after each node completes."""
    result = initial_state
    with live_placeholder.container():
        st.markdown(f"#### :material/timeline: Live agent trace — `{run_label}`")
        st.caption("Starting pipeline…")
    try:
        for chunk in graph.stream(initial_state, stream_mode="values"):
            result = chunk
            with live_placeholder.container():
                st.markdown(f"#### :material/timeline: Live agent trace — `{run_label}`")
                write_trace_steps(chunk.get("trace") or [])
    except Exception as e:
        log_app_error(
            error=e,
            context="run_pipeline_stream",
            filename=str(initial_state.get("filename") or run_label),
            extra={"run_label": run_label[:120]},
        )
        trace = list(result.get("trace") or [])
        trace.append(
            {
                "node": "runtime",
                "contribution": "Pipeline terminated unexpectedly; preserving partial progress.",
                "detail": str(e),
                "duration_ms": None,
            }
        )
        result = {
            **result,
            "error": result.get("error") or f"Pipeline interrupted: {str(e)}",
            "trace": trace,
        }
        with live_placeholder.container():
            st.markdown(f"#### :material/timeline: Live agent trace — `{run_label}`")
            write_trace_steps(result.get("trace") or [])
    return result


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');

    :root {
        --ra-bg: #0b1020;
        --ra-bg-soft: rgba(15, 23, 42, 0.76);
        --ra-surface: rgba(15, 23, 42, 0.72);
        --ra-surface-strong: rgba(15, 23, 42, 0.9);
        --ra-border: rgba(148, 163, 184, 0.18);
        --ra-border-strong: rgba(96, 165, 250, 0.28);
        --ra-text: #e5eefb;
        --ra-text-soft: #9fb2cf;
        --ra-accent: #7c3aed;
        --ra-accent-2: #38bdf8;
        --ra-success: #34d399;
        --ra-danger: #fb7185;
        --ra-warning: #fbbf24;
    }

    html, body, [class*="css"], [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.03em;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(124, 58, 237, 0.18), transparent 30%),
            radial-gradient(circle at top right, rgba(56, 189, 248, 0.15), transparent 24%),
            linear-gradient(180deg, #08101f 0%, #0b1020 100%);
        color: var(--ra-text);
    }
    .block-container {
        max-width: 1280px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    .ra-hero {
        position: relative;
        overflow: hidden;
        margin-bottom: 1.4rem;
        padding: 1.7rem 1.8rem;
        border: 1px solid var(--ra-border-strong);
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.94), rgba(15, 23, 42, 0.82));
        box-shadow: 0 30px 80px rgba(2, 6, 23, 0.45);
    }
    .ra-hero::after {
        content: "";
        position: absolute;
        inset: auto -4rem -4rem auto;
        width: 14rem;
        height: 14rem;
        border-radius: 999px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.26), transparent 65%);
        pointer-events: none;
    }
    .ra-eyebrow {
        margin: 0 0 0.3rem 0;
        color: #8fb3ff;
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.16em;
    }
    .ra-hero-title, .ra-section-title, .ra-summary-title, .ra-panel-title {
        margin: 0;
        color: #f8fbff;
    }
    .ra-hero-title {
        font-size: clamp(2rem, 3vw, 3.2rem);
        line-height: 1.02;
        max-width: 12ch;
    }
    .ra-hero-copy, .ra-section-copy, .ra-panel-copy, .ra-reason-copy {
        color: var(--ra-text-soft);
        line-height: 1.6;
    }
    .ra-hero-copy {
        max-width: 52rem;
        margin: 0.75rem 0 0 0;
        font-size: 1rem;
    }
    .ra-section-intro {
        margin: 0.35rem 0 1.15rem 0;
    }
    .ra-section-title {
        font-size: 1.25rem;
        margin-bottom: 0.35rem;
    }
    .ra-section-copy, .ra-panel-copy {
        margin: 0;
        font-size: 0.97rem;
    }

    .ra-stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1rem 0 1.35rem 0;
    }
    .ra-stat-card, .ra-info-panel, .ra-summary-card {
        background: var(--ra-surface);
        border: 1px solid var(--ra-border);
        border-radius: 20px;
        box-shadow: 0 18px 50px rgba(2, 6, 23, 0.25);
        backdrop-filter: blur(14px);
    }
    .ra-stat-card {
        padding: 1rem 1.05rem;
    }
    .ra-stat-label, .ra-score-label {
        color: #8ea4c4;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }
    .ra-stat-value {
        margin-top: 0.35rem;
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fbff;
    }
    .ra-stat-hint {
        margin-top: 0.25rem;
        color: var(--ra-text-soft);
        font-size: 0.84rem;
    }

    .ra-info-panel {
        padding: 1.2rem 1.25rem;
        margin: 0.6rem 0 1rem 0;
    }
    .ra-panel-title {
        font-size: 1.05rem;
        margin-bottom: 0.45rem;
    }
    .ra-info-list {
        margin: 0.9rem 0 0 1rem;
        padding: 0;
        color: var(--ra-text-soft);
    }
    .ra-info-list li {
        margin-bottom: 0.35rem;
    }

    .ra-badge-row, .ra-meta-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
    }
    .ra-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.32rem;
        padding: 0.48rem 0.72rem;
        border-radius: 999px;
        border: 1px solid var(--ra-border);
        background: rgba(15, 23, 42, 0.9);
        color: #dbeafe;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .ra-badge--positive {
        border-color: rgba(52, 211, 153, 0.35);
        background: rgba(6, 78, 59, 0.35);
        color: #9ff3ca;
    }
    .ra-badge--negative {
        border-color: rgba(251, 113, 133, 0.32);
        background: rgba(127, 29, 29, 0.3);
        color: #fecdd3;
    }
    .ra-badge--warning {
        border-color: rgba(251, 191, 36, 0.36);
        background: rgba(120, 53, 15, 0.28);
        color: #fde68a;
    }
    .ra-badge--neutral {
        border-color: rgba(148, 163, 184, 0.24);
        color: #dbeafe;
    }

    .ra-summary-card {
        padding: 1.2rem 1.2rem 1.1rem 1.2rem;
        margin: 0.25rem 0 1.2rem 0;
    }
    .ra-summary-head {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .ra-summary-title {
        font-size: 1.35rem;
        line-height: 1.18;
    }
    .ra-summary-grid {
        display: grid;
        grid-template-columns: minmax(180px, 230px) minmax(0, 1fr);
        gap: 0.9rem;
    }
    .ra-score-card, .ra-reason-card {
        padding: 1rem;
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.12);
        background: rgba(15, 23, 42, 0.72);
    }
    .ra-score-value {
        display: flex;
        align-items: baseline;
        gap: 0.3rem;
        margin: 0.4rem 0 0.9rem 0;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.3rem;
        font-weight: 700;
        color: #f8fbff;
    }
    .ra-score-value span {
        font-size: 0.95rem;
        color: var(--ra-text-soft);
    }
    .ra-reason-copy {
        margin: 0.45rem 0 0 0;
        font-size: 0.96rem;
    }
    .ra-meta-row {
        margin-top: 1rem;
    }
    .ra-meta-chip {
        display: inline-flex;
        gap: 0.45rem;
        align-items: center;
        padding: 0.5rem 0.7rem;
        border-radius: 14px;
        background: rgba(30, 41, 59, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.14);
        color: var(--ra-text-soft);
        font-size: 0.82rem;
    }
    .ra-meta-chip strong {
        color: #f8fbff;
        font-weight: 600;
    }

    .ra-progress {
        height: 10px;
        overflow: hidden;
        border-radius: 999px;
        background: rgba(51, 65, 85, 0.72);
    }
    .ra-progress-bar {
        height: 100%;
        border-radius: inherit;
    }
    .ra-progress-bar--positive {
        background: linear-gradient(90deg, #34d399, #22c55e);
    }
    .ra-progress-bar--negative {
        background: linear-gradient(90deg, #fb7185, #ef4444);
    }
    .ra-progress-bar--warning {
        background: linear-gradient(90deg, #fbbf24, #f59e0b);
    }

    .stButton>button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white;
        border: 1px solid rgba(191, 219, 254, 0.12);
        border-radius: 999px;
        padding: 0.68rem 1.2rem;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        box-shadow: 0 12px 28px rgba(79, 70, 229, 0.35);
    }
    .stButton>button:hover {
        border-color: rgba(191, 219, 254, 0.22);
        background: linear-gradient(135deg, #8b5cf6, #6366f1);
    }

    .stTextArea textarea,
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] input,
    div[data-baseweb="base-input"] input {
        background: rgba(15, 23, 42, 0.76) !important;
        color: #eef4ff !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        border-radius: 16px !important;
    }
    .stTextArea {
        margin-bottom: 0.85rem;
    }
    .stTextArea textarea:focus,
    div[data-baseweb="input"] input:focus,
    div[data-baseweb="base-input"] input:focus {
        border-color: rgba(96, 165, 250, 0.52) !important;
        box-shadow: 0 0 0 1px rgba(96, 165, 250, 0.18) !important;
    }
    div[data-testid="stFileUploaderDropzone"] {
        background: rgba(15, 23, 42, 0.72);
        border: 1px dashed rgba(96, 165, 250, 0.32);
        border-radius: 22px;
        padding: 1.15rem 1rem;
    }
    div[role="radiogroup"] {
        gap: 0.6rem;
    }
    div[role="radiogroup"] label {
        background: rgba(15, 23, 42, 0.76);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
    }
    div[data-baseweb="tab-list"] {
        gap: 0.45rem;
        margin: 0.55rem 0 0.9rem 0;
    }
    button[role="tab"] {
        border-radius: 999px !important;
        padding: 0.4rem 0.85rem !important;
        background: rgba(15, 23, 42, 0.72) !important;
        border: 1px solid rgba(148, 163, 184, 0.16) !important;
    }
    button[role="tab"][aria-selected="true"] {
        background: rgba(59, 130, 246, 0.16) !important;
        border-color: rgba(96, 165, 250, 0.34) !important;
    }
    details[data-testid="stExpander"] {
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        border-radius: 22px !important;
        background: rgba(15, 23, 42, 0.52) !important;
        overflow: hidden;
        margin: 0 0 1.05rem 0;
    }
    details[data-testid="stExpander"] summary {
        padding-top: 0.2rem;
        padding-bottom: 0.2rem;
        font-size: 1rem;
        font-weight: 600;
        color: #f8fbff;
    }
    div[data-testid="stStatusWidget"] {
        border-radius: 22px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: rgba(15, 23, 42, 0.72);
    }
    .material-sym {
        font-family: 'Material Symbols Outlined';
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        font-size: 1.2rem;
        vertical-align: -0.22em;
        margin-right: 0.35rem;
        user-select: none;
    }

    @media (max-width: 900px) {
        .ra-summary-head, .ra-summary-grid {
            grid-template-columns: 1fr;
            display: grid;
        }
        .ra-summary-head {
            gap: 0.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
<style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state init ────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = []
if "research_focus" not in st.session_state:
    st.session_state.research_focus = ""
if "discovery_results" not in st.session_state:
    st.session_state.discovery_results = []
if "llm_enabled" not in st.session_state:
    st.session_state.llm_enabled = True
if "usage_open_logged" not in st.session_state:
    st.session_state.usage_open_logged = False
if not st.session_state.usage_open_logged:
    log_usage_event("app_page_open", {"page": "research_workspace"})
    st.session_state.usage_open_logged = True

# ── Main area ─────────────────────────────────────────────────────────────────
total = len(st.session_state.results)
fits = sum(1 for r in st.session_state.results if r.get("fit"))
d_total = len(st.session_state.discovery_results)

intro_col, utility_col = st.columns([1.55, 0.95], gap="large")
with intro_col:
    st.markdown(
        _build_section_intro_html(
            "Topic setup",
            "Start with a precise research focus",
            "A tighter topic gives the agent better summaries, stronger qualification calls, and cleaner evidence matrices.",
        ),
        unsafe_allow_html=True,
    )
with utility_col:
    info_items = [
        "Upload PDFs when you already have candidate papers.",
        "Use discovery when you want the agent to search and qualify works.",
        "Trace views remain available for every completed run.",
    ]
    if total or d_total:
        info_items[0] = "The workspace already contains saved review results."
    st.markdown(
        _build_info_panel_html(
            "How this workspace is organized",
            "The layout is now split into a clearer input area, action controls, and review-ready result cards.",
            info_items,
        ),
        unsafe_allow_html=True,
    )
    if total or d_total:
        if st.button("Clear all results", icon=":material/delete_sweep:"):
            st.session_state.results = []
            st.session_state.discovery_results = []
            st.rerun()

topic_input = st.text_area(
    "What is your research topic?",
    placeholder="e.g. Machine learning approaches to credit risk scoring using XGBoost and SHAP explainability in low-data environments...",
    height=110,
)
st.session_state.research_focus = (topic_input or "").strip()
st.toggle(
    "Use LLM (disable for deterministic fallback mode)",
    key="llm_enabled",
    help="When off, all LLM-dependent steps use deterministic rule-based fallbacks.",
)

st.markdown(
    _build_section_intro_html(
        "Workflow mode",
        "Choose how you want to evaluate evidence",
        "Switch between scoring your own PDFs and letting the agent discover qualified journal works for the topic.",
    ),
    unsafe_allow_html=True,
)
has_papers = st.radio(
    "Do you already have PDFs/journals to score?",
    ["Yes", "No"],
    horizontal=True,
)

if has_papers == "Yes":
    st.markdown(
        _build_section_intro_html(
            "PDF scoring",
            "Upload papers for direct fit scoring",
            "Each paper gets a score, a decision badge, a structured evidence matrix, and a full agent trace.",
        ),
        unsafe_allow_html=True,
    )
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and st.session_state.research_focus:
        already_processed = {r["filename"] for r in st.session_state.results}
        new_files = [f for f in uploaded_files if f.name not in already_processed]

        if new_files:
            st.markdown(
                _build_stats_grid_html(
                    [
                        ("Selected files", str(len(uploaded_files)), "Current upload batch"),
                        ("New to process", str(len(new_files)), "Ready for scoring"),
                        ("Already scored", str(len(uploaded_files) - len(new_files)), "Skipped duplicates"),
                    ]
                ),
                unsafe_allow_html=True,
            )
            if st.button(f"Score {len(new_files)} paper(s)", type="primary", icon=":material/play_arrow:"):
                progress = st.progress(0)
                status = st.empty()

                for i, uploaded_file in enumerate(new_files):
                    status.markdown(f":material/hourglass_empty: Processing **{uploaded_file.name}**...")

                    try:
                        pdf_bytes = uploaded_file.read()
                        pdf_text = extract_text_from_pdf(pdf_bytes)

                        initial_state = {
                            "filename": uploaded_file.name,
                            "pdf_text": pdf_text,
                            "summary": "",
                            "key_findings": "",
                            "methodology": "",
                            "relevance_score": 0.0,
                            "relevance_reason": "",
                            "source_profile": {},
                            "fit": False,
                            "error": None,
                            "llm_enabled": bool(st.session_state.get("llm_enabled", True)),
                            "llm_used": False,
                            "fallback_reason": "",
                            "trace": [],
                        }

                        with st.status(
                            f"Agent pipeline — `{uploaded_file.name}`",
                            expanded=True,
                        ) as run_status:
                            live_trace = st.empty()
                            result = run_pipeline_stream(pipeline, initial_state, live_trace, uploaded_file.name)
                            if result.get("error"):
                                run_status.update(
                                    label=f":material/warning: Interrupted `{uploaded_file.name}` (partial results kept)",
                                    state="error",
                                    expanded=True,
                                )
                            else:
                                run_status.update(
                                    label=f":material/check_circle: Finished `{uploaded_file.name}`",
                                    state="complete",
                                    expanded=False,
                                )
                        focus = st.session_state.get("research_focus") or ""
                        trace_id = persist_pipeline_run(
                            {
                                "filename": result.get("filename"),
                                "research_focus": focus,
                                "trace": result.get("trace") or [],
                                "relevance_score": result.get("relevance_score"),
                                "fit": result.get("fit"),
                                "error": result.get("error"),
                                "llm_used": result.get("llm_used"),
                                "fallback_reason": result.get("fallback_reason"),
                            }
                        )
                        if trace_id:
                            result["trace_id"] = trace_id
                        st.session_state.results.append(result)

                    except Exception as e:
                        log_app_error(
                            error=e,
                            context="pdf_upload_process",
                            filename=uploaded_file.name,
                            extra={"research_focus": (st.session_state.get("research_focus") or "")[:120]},
                        )
                        st.session_state.results.append(
                            {
                                "filename": uploaded_file.name,
                                "error": str(e),
                                "fit": False,
                            }
                        )

                    progress.progress((i + 1) / len(new_files))

                status.empty()
                progress.empty()
                st.success(":material/check_circle: Done! Results below.")
                st.rerun()
    elif uploaded_files and not st.session_state.research_focus:
        st.warning("Please add your research topic first.")
else:
    st.markdown(
        _build_section_intro_html(
            "Discovery",
            "Let the agent search and qualify journals",
            "Discovery runs evaluate candidate papers in rounds and keep only the strongest topical and scholarly matches.",
        ),
        unsafe_allow_html=True,
    )
    config_col1, config_col2 = st.columns(2)
    with config_col1:
        max_rounds = st.number_input("Max rounds", min_value=1, max_value=12, value=5)
    with config_col2:
        batch_size = st.number_input("Candidates per round", min_value=3, max_value=20, value=8)
    target_qualified_count = st.number_input(
        "Target qualified papers",
        min_value=3,
        max_value=10,
        value=3,
        help="Discovery stops once this many qualified papers are found.",
    )
    st.markdown(
        _build_stats_grid_html(
            [
                ("Round limit", str(int(max_rounds)), "How many search passes to allow"),
                ("Candidates / round", str(int(batch_size)), "Papers screened each pass"),
                ("Target qualified", str(int(target_qualified_count)), "Current stop threshold"),
            ]
        ),
        unsafe_allow_html=True,
    )
    can_hunt = bool(st.session_state.research_focus)
    if st.button("Hunt journals for me", type="primary", disabled=not can_hunt, icon=":material/play_arrow:"):
        topic = st.session_state.research_focus
        initial_state = {
            "filename": f"discovery:{topic}",
            "pdf_text": "",
            "summary": "",
            "key_findings": "",
            "methodology": "",
            "relevance_score": 0.0,
            "relevance_reason": "",
            "fit": False,
            "error": None,
            "llm_enabled": bool(st.session_state.get("llm_enabled", True)),
            "llm_used": False,
            "fallback_reason": "",
            "trace": [],
            "topic": topic,
            "discovery_batch_size": int(batch_size),
            "max_discovery_rounds": int(max_rounds),
            "target_qualified_count": int(target_qualified_count),
            "qualified_works": [],
            "evaluated_candidates": [],
            "discovered_candidates": [],
            "candidate_source_profile": {},
        }
        with st.status("Discovery pipeline", expanded=True) as run_status:
            live_trace = st.empty()
            result = run_pipeline_stream(discovery_pipeline, initial_state, live_trace, f"topic:{topic}")
            if result.get("error"):
                run_status.update(
                    label=":material/warning: Topic agent interrupted (partial results kept)",
                    state="error",
                    expanded=True,
                )
            else:
                run_status.update(label=":material/check_circle: Topic agent finished", state="complete", expanded=False)
        st.session_state.discovery_results.append(result)
        st.rerun()
    if not can_hunt:
        st.caption("Add your research topic to enable hunting.")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.results:
    st.divider()
    st.markdown(
        _build_section_intro_html(
            "Reviewed papers",
            "Scored PDF results",
            "Results are sorted by relevance score so the strongest matches surface first.",
        ),
        unsafe_allow_html=True,
    )

    # Sort: relevant first
    sorted_results = sorted(
        st.session_state.results,
        key=lambda x: x.get("relevance_score", 0),
        reverse=True,
    )

    for result in sorted_results:
        filename = result.get("filename", "Unknown")
        error = result.get("error")
        score = float(result.get("relevance_score", 0) or 0.0)
        fit = bool(result.get("fit", False))
        score_tone = "positive" if fit else "negative"
        status_label = "Relevant" if fit else "Not relevant"
        expander_label = f"{status_label} · {score:.2f} · {filename}"

        with st.expander(f":material/description: {expander_label}", expanded=False):
            if error:
                st.error(f":material/error: Error: {error}")
                continue

            st.markdown(
                _build_summary_card_html(
                    title=filename,
                    eyebrow="PDF evaluation",
                    badges=[
                        _build_badge_html(status_label, score_tone, "check_circle" if fit else "cancel"),
                        _build_badge_html(
                            "LLM used" if result.get("llm_used") else "No-LLM fallback",
                            "neutral",
                            "smart_toy" if result.get("llm_used") else "rule",
                        ),
                        _build_badge_html("Trace available", "neutral", "account_tree"),
                    ],
                    score=score,
                    score_label="Relevance score",
                    reason_label="Decision rationale",
                    reason=result.get("relevance_reason", ""),
                    metadata=[
                        ("Summary", "Ready" if result.get("summary") else "Missing"),
                        ("Matrix", "Ready" if result.get("source_profile") else "Missing"),
                            ("Use examples", "Ready" if result.get("citation_use_examples") else "Missing"),
                    ],
                    tone=score_tone,
                ),
                unsafe_allow_html=True,
            )

            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
                [
                    ":material/notes: Summary",
                    ":material/vpn_key: Key Findings",
                    ":material/build: Methodology",
                    ":material/table_chart: Evidence Matrix",
                    ":material/report: Risk flags",
                    ":material/lightbulb: Citation-use examples",
                    ":material/fact_check: Evidence contract",
                    ":material/account_tree: Agent trace",
                ]
            )

            with tab1:
                st.markdown(result.get("summary", "Not available."))

            with tab2:
                st.markdown(result.get("key_findings", "Not available."))

            with tab3:
                st.markdown(result.get("methodology", "Not available."))

            with tab4:
                _render_source_matrix(dict(result.get("source_profile") or {}))

            with tab5:
                _render_risk_flags(list(result.get("risk_flags") or []))

            with tab6:
                _render_citation_use_examples(list(result.get("citation_use_examples") or []))

            with tab7:
                _render_evidence_contract(dict(result.get("evidence_contract") or {}))

            with tab8:
                trace = result.get("trace") or []
                oid = result.get("trace_id")
                if oid:
                    st.caption(f"Stored run id: `{oid}` (MongoDB)")
                fb = (result.get("fallback_reason") or "").strip()
                if fb:
                    st.caption(f"Fallback reason: {fb}")
                st.markdown("**Flowchart**")
                write_trace_flowchart(trace)
                st.markdown("**Step-by-step trace**")
                write_trace_steps(trace)

if st.session_state.discovery_results:
    st.divider()
    st.markdown(
        _build_section_intro_html(
            "Discovery runs",
            "Qualified journal matches",
            "Each run keeps the strongest candidates together with the evidence matrix, abstract, links, and underlying trace.",
        ),
        unsafe_allow_html=True,
    )
    for i, run in enumerate(reversed(st.session_state.discovery_results), start=1):
        topic = run.get("topic", "Unknown topic")
        qualified = run.get("qualified_works") or []
        err = run.get("error")
        with st.expander(
            f":material/manage_search: Run {i} · {len(qualified)} qualified · {topic}",
            expanded=False,
        ):
            if err:
                st.error(err)
                continue
            st.markdown(
                _build_stats_grid_html(
                    [
                        ("Qualified works", str(len(qualified)), "Works kept after evaluation"),
                        ("Topic", topic[:36] + ("..." if len(topic) > 36 else ""), "Current research focus"),
                        ("Trace", "Available", "Discovery pipeline history"),
                    ]
                ),
                unsafe_allow_html=True,
            )
            if not qualified:
                st.info("No qualified works found within current limits.")
            visible_qualified = int(run.get("target_qualified_count") or 3)
            for idx, item in enumerate(qualified[:visible_qualified], start=1):
                score = float(item.get("score") or 0.0)
                fit = bool(item.get("fit"))
                quality = bool(item.get("quality"))
                tone = "positive" if fit and quality else "warning"
                st.markdown(
                    _build_summary_card_html(
                        title=f"{idx}. {item.get('title', 'Untitled')}",
                        eyebrow="Qualified discovery match",
                        badges=[
                            _build_badge_html(
                                "Topic fit" if fit else "Topic mismatch",
                                "positive" if fit else "negative",
                                "check_circle" if fit else "cancel",
                            ),
                            _build_badge_html(
                                "Scholarly quality" if quality else "Quality risk",
                                "positive" if quality else "warning",
                                "verified" if quality else "report",
                            ),
                        ],
                        score=score,
                        score_label="Qualification score",
                        reason_label="Why this qualified",
                        reason=item.get("reason", "No rationale returned."),
                        metadata=[
                            ("Venue", str(item.get("venue", "Unknown venue"))),
                            ("Year", str(item.get("year", "n/a"))),
                            ("Citations", str(item.get("cited_by_count", 0))),
                        ],
                        tone=tone,
                    ),
                    unsafe_allow_html=True,
                )

                tab_overview, tab_matrix, tab_risk, tab_abstract, tab_links, tab_use, tab_contract = st.tabs(
                    [
                        ":material/push_pin: Overview",
                        ":material/table_chart: Evidence Matrix",
                        ":material/report: Risk flags",
                        ":material/article: Abstract",
                        ":material/link: Sources",
                        ":material/lightbulb: Citation-use examples",
                        ":material/fact_check: Evidence contract",
                    ]
                )
                with tab_overview:
                    st.markdown(f"**Venue:** {item.get('venue', 'Unknown venue')}")
                    st.markdown(f"**Year:** {item.get('year', 'n/a')}")
                    st.markdown(f"**Citations:** {item.get('cited_by_count', 0)}")
                    st.markdown(f"**Topical fit:** {'YES' if fit else 'NO'}")
                    st.markdown(f"**Scholarly quality:** {'YES' if quality else 'NO'}")
                with tab_matrix:
                    _render_source_matrix(dict(item.get("source_profile") or {}))
                with tab_risk:
                    _render_risk_flags(list(item.get("risk_flags") or []))
                with tab_abstract:
                    st.markdown(item.get("abstract") or "Abstract not available.")
                with tab_links:
                    if item.get("doi"):
                        st.markdown(f"DOI: `{item.get('doi')}`")
                    if item.get("url"):
                        st.markdown(f"OpenAlex: {item.get('url')}")
                    if not item.get("doi") and not item.get("url"):
                        st.caption("No external links available for this work.")
                with tab_use:
                    _render_citation_use_examples(list(item.get("citation_use_examples") or []))
                with tab_contract:
                    _render_evidence_contract(dict(item.get("evidence_contract") or {}))
                st.markdown("---")
            st.markdown("#### :material/account_tree: Agent trace")
            trace = run.get("trace") or []
            write_trace_flowchart(trace)
            write_trace_steps(trace)

all_chat_docs = _build_chat_context_docs()
if all_chat_docs:
    st.divider()
    st.markdown(
        _build_section_intro_html(
            "Interactive analysis",
            "Chat with selected papers",
            "Ask follow-up questions grounded in selected papers. Answers include citation anchors to source titles/files.",
        ),
        unsafe_allow_html=True,
    )
    source_options = [d["source"] for d in all_chat_docs]
    selected_sources = st.multiselect(
        "Select paper sources for chat grounding",
        options=source_options,
        default=source_options[: min(3, len(source_options))],
    )
    if selected_sources:
        docs_by_source = {d["source"]: d for d in all_chat_docs}
        selected_docs = [docs_by_source[s] for s in selected_sources if s in docs_by_source]
        focus = str(st.session_state.get("research_focus") or "")
        chat_key = _chat_state_key(selected_sources, focus)
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []
        messages = st.session_state[chat_key]
        st.caption(f"Chat context: {len(selected_docs)} source(s) · focus: {focus or 'N/A'}")
        for m in messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("meta"):
                    st.caption(m["meta"])
        user_q = st.chat_input("Ask a grounded question about selected papers")
        if user_q:
            messages.append({"role": "user", "content": user_q, "meta": ""})
            with st.chat_message("user"):
                st.markdown(user_q)
            with st.chat_message("assistant"):
                status_line = st.empty()
                status_line.caption(":material/schedule: Preparing grounded answer...")
                llm_used_chat = False
                fallback = ""
                answer = ""
                try:
                    stream_iter, llm_used_chat, fallback = _answer_paper_chat_stream(
                        user_q,
                        selected_docs,
                        focus,
                        bool(st.session_state.get("llm_enabled", True)),
                    )
                    status_line.caption(":material/sync: Generating response...")
                    answer = st.write_stream(stream_iter)
                    if not isinstance(answer, str):
                        answer = str(answer or "").strip()
                    if not answer:
                        answer, llm_used_chat, fallback = _answer_paper_chat(
                            user_q,
                            selected_docs,
                            focus,
                            bool(st.session_state.get("llm_enabled", True)),
                        )
                        st.markdown(answer)
                except Exception as e:
                    log_app_error(
                        error=e,
                        context="chat_answer",
                        extra={"question_len": len(user_q), "source_count": len(selected_docs)},
                    )
                    answer = "Sorry, chat response failed. Please try again."
                    fallback = f"chat_error: {str(e)}"
                    st.error(answer)
                meta = "LLM used" if llm_used_chat else (fallback or "deterministic fallback")
                log_usage_event(
                    "chat_interaction",
                    {
                        "question_len": len(user_q),
                        "answer_len": len(answer or ""),
                        "source_count": len(selected_docs),
                        "llm_used": bool(llm_used_chat),
                        "fallback_reason": (fallback or "")[:160],
                    },
                )
                status_line.caption(":material/check_circle: Response ready")
                st.caption(meta)
            messages.append({"role": "assistant", "content": answer, "meta": meta})
            st.session_state[chat_key] = messages
            st.rerun()
    else:
        st.info("Select at least one source to enable chat.")
