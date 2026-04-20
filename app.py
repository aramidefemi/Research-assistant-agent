from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from typing import Any
from utils.pdf_reader import extract_text_from_pdf
from utils.trace_store import persist_pipeline_run
from graph.pipeline import pipeline, discovery_pipeline


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
        st.caption(f"LLM-timed steps total ≈ **{total_ms:.0f} ms**")
    for i, step in enumerate(trace, start=1):
        node = step.get("node", "?")
        with st.expander(f"Step {i} — {node}", expanded=(i <= 2)):
            st.markdown(step.get("contribution", ""))
            det = step.get("detail") or ""
            if det:
                st.caption(det)
            dm = step.get("duration_ms")
            if dm is not None:
                st.caption(f"⏱ {dm:.0f} ms")
            result = step.get("result")
            if result:
                with st.expander("Stage result", expanded=False):
                    _render_stage_result(result, "")


def run_pipeline_stream(graph, initial_state: dict, live_placeholder, run_label: str) -> dict:
    """Run a graph with streaming; updates `live_placeholder` after each node completes."""
    result = initial_state
    with live_placeholder.container():
        st.markdown(f"#### 🧭 Live agent trace — `{run_label}`")
        st.caption("Starting pipeline…")
    try:
        for chunk in graph.stream(initial_state, stream_mode="values"):
            result = chunk
            with live_placeholder.container():
                st.markdown(f"#### 🧭 Live agent trace — `{run_label}`")
                write_trace_steps(chunk.get("trace") or [])
    except Exception as e:
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
            st.markdown(f"#### 🧭 Live agent trace — `{run_label}`")
            write_trace_steps(result.get("trace") or [])
    return result


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

    .stApp { background-color: #0f0f11; color: #e8e8e8; }

    .metric-card {
        background: #1a1a1f;
        border: 1px solid #2a2a35;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .fit-yes {
        background: #0d2b1a;
        border: 1px solid #1a6b3a;
        border-radius: 8px;
        padding: 1rem;
        color: #4ade80;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .fit-no {
        background: #2b0d0d;
        border: 1px solid #6b1a1a;
        border-radius: 8px;
        padding: 1rem;
        color: #f87171;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .score-bar-wrap {
        background: #1a1a1f;
        border-radius: 4px;
        height: 8px;
        margin-top: 6px;
    }
    .stButton>button {
        background: #4f46e5;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.4rem;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
    }
    .stButton>button:hover { background: #6366f1; }
    .sidebar-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.4rem;
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
if "evaluation_depth" not in st.session_state:
    st.session_state.evaluation_depth = "full"
if "discovery_results" not in st.session_state:
    st.session_state.discovery_results = []

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("## Research Agent")
st.markdown("Start with your research topic, then either upload papers to score or let the agent hunt for journals.")

st.radio(
    "Evaluation depth",
    ["full", "quick"],
    format_func=lambda x: (
        "Full — score, fit, and why (2 LLM steps)" if x == "full" else "Quick — score & fit only (1 step)"
    ),
    key="evaluation_depth",
)

if st.session_state.results or st.session_state.discovery_results:
    total = len(st.session_state.results)
    fits = sum(1 for r in st.session_state.results if r.get("fit"))
    d_total = len(st.session_state.discovery_results)
    st.caption(f"Papers processed: {total} | Relevant: {fits}/{total} | Discovery runs: {d_total}")
    if st.button("🗑 Clear all results"):
        st.session_state.results = []
        st.session_state.discovery_results = []
        st.rerun()

topic_input = st.text_area(
    "What is your research topic?",
    placeholder="e.g. Machine learning approaches to credit risk scoring using XGBoost and SHAP explainability in low-data environments...",
    height=110,
)
st.session_state.research_focus = (topic_input or "").strip()

has_papers = st.radio(
    "Do you already have PDFs/journals to score?",
    ["Yes", "No"],
    horizontal=True,
)

if has_papers == "Yes":
    st.markdown("Upload one or more PDFs and the agent will score fit against your topic.")
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
            st.markdown(f"**{len(new_files)} new paper(s) to process.**")
            if st.button(f"▶ Score {len(new_files)} paper(s)", type="primary"):
                progress = st.progress(0)
                status = st.empty()

                for i, uploaded_file in enumerate(new_files):
                    status.markdown(f"⏳ Processing **{uploaded_file.name}**...")

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
                            "fit": False,
                            "error": None,
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
                                    label=f"⚠️ Interrupted `{uploaded_file.name}` (partial results kept)",
                                    state="error",
                                    expanded=True,
                                )
                            else:
                                run_status.update(
                                    label=f"✅ Finished `{uploaded_file.name}`",
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
                            }
                        )
                        if trace_id:
                            result["trace_id"] = trace_id
                        st.session_state.results.append(result)

                    except Exception as e:
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
                st.success("✅ Done! Results below.")
                st.rerun()
    elif uploaded_files and not st.session_state.research_focus:
        st.warning("Please add your research topic first.")
else:
    st.markdown("No papers yet? Let the agent hunt and qualify top journal works for your topic.")
    max_rounds = st.number_input("Max rounds", min_value=1, max_value=12, value=5)
    batch_size = st.number_input("Candidates per round", min_value=3, max_value=20, value=8)
    can_hunt = bool(st.session_state.research_focus)
    if st.button("▶ Hunt journals for me", type="primary", disabled=not can_hunt):
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
            "trace": [],
            "topic": topic,
            "discovery_batch_size": int(batch_size),
            "max_discovery_rounds": int(max_rounds),
            "target_qualified_count": 2,
            "qualified_works": [],
            "evaluated_candidates": [],
            "discovered_candidates": [],
        }
        with st.status("Discovery pipeline", expanded=True) as run_status:
            live_trace = st.empty()
            result = run_pipeline_stream(discovery_pipeline, initial_state, live_trace, f"topic:{topic}")
            if result.get("error"):
                run_status.update(label="⚠️ Topic agent interrupted (partial results kept)", state="error", expanded=True)
            else:
                run_status.update(label="✅ Topic agent finished", state="complete", expanded=False)
        st.session_state.discovery_results.append(result)
        st.rerun()
    if not can_hunt:
        st.caption("Add your research topic to enable hunting.")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.results:
    st.markdown("---")
    st.markdown("## Results")

    # Sort: relevant first
    sorted_results = sorted(
        st.session_state.results,
        key=lambda x: x.get("relevance_score", 0),
        reverse=True,
    )

    for result in sorted_results:
        filename = result.get("filename", "Unknown")
        error = result.get("error")

        with st.expander(f"📄 {filename}", expanded=False):
            if error:
                st.error(f"❌ Error: {error}")
                continue

            score = result.get("relevance_score", 0)
            fit = result.get("fit", False)

            col1, col2 = st.columns([1, 3])

            with col1:
                fit_html = (
                    '<div class="fit-yes">✅ RELEVANT</div>'
                    if fit else
                    '<div class="fit-no">❌ NOT RELEVANT</div>'
                )
                st.markdown(fit_html, unsafe_allow_html=True)
                st.markdown(f"**Relevance score:** `{score:.2f}`")
                score_pct = int(score * 100)
                st.markdown(
                    f'<div class="score-bar-wrap"><div style="background:{"#4ade80" if fit else "#f87171"};'
                    f'width:{score_pct}%;height:8px;border-radius:4px;"></div></div>',
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(f"**Why:** {result.get('relevance_reason', '')}")

            st.markdown("---")

            tab1, tab2, tab3, tab4 = st.tabs(
                ["📝 Summary", "🔑 Key Findings", "🔧 Methodology", "🧭 Agent trace"]
            )

            with tab1:
                st.markdown(result.get("summary", "Not available."))

            with tab2:
                st.markdown(result.get("key_findings", "Not available."))

            with tab3:
                st.markdown(result.get("methodology", "Not available."))

            with tab4:
                trace = result.get("trace") or []
                oid = result.get("trace_id")
                if oid:
                    st.caption(f"Stored run id: `{oid}` (MongoDB)")
                write_trace_steps(trace)

if st.session_state.discovery_results:
    st.markdown("---")
    st.markdown("## Discovery Results")
    for i, run in enumerate(reversed(st.session_state.discovery_results), start=1):
        topic = run.get("topic", "Unknown topic")
        qualified = run.get("qualified_works") or []
        err = run.get("error")
        with st.expander(f"🔎 Run {i} — {topic}", expanded=False):
            if err:
                st.error(err)
                continue
            st.markdown(f"**Qualified works:** {len(qualified)}")
            if not qualified:
                st.info("No qualified works found within current limits.")
            for idx, item in enumerate(qualified[:2], start=1):
                st.markdown(f"### {idx}. {item.get('title', 'Untitled')}")
                meta = f"{item.get('venue', 'Unknown venue')} | {item.get('year', 'n/a')} | citations: {item.get('cited_by_count', 0)}"
                st.caption(meta)

                score = float(item.get("score") or 0.0)
                fit = bool(item.get("fit"))
                quality = bool(item.get("quality"))
                score_pct = max(0, min(100, int(score * 100)))
                fit_badge = (
                    '<div class="fit-yes">✅ TOPIC FIT</div>'
                    if fit else
                    '<div class="fit-no">❌ TOPIC MISMATCH</div>'
                )
                quality_badge = (
                    '<div class="fit-yes">✅ SCHOLARLY QUALITY</div>'
                    if quality else
                    '<div class="fit-no">❌ QUALITY RISK</div>'
                )

                c1, c2 = st.columns([1, 3])
                with c1:
                    st.markdown(fit_badge, unsafe_allow_html=True)
                    st.markdown(quality_badge, unsafe_allow_html=True)
                    st.markdown(f"**Score:** `{score:.2f}`")
                    st.markdown(
                        f'<div class="score-bar-wrap"><div style="background:#4ade80;'
                        f'width:{score_pct}%;height:8px;border-radius:4px;"></div></div>',
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(f"**Why this qualified:** {item.get('reason', 'No rationale returned.')}")

                tab_overview, tab_abstract, tab_links = st.tabs(
                    ["📌 Overview", "🧾 Abstract", "🔗 Sources"]
                )
                with tab_overview:
                    st.markdown(f"**Venue:** {item.get('venue', 'Unknown venue')}")
                    st.markdown(f"**Year:** {item.get('year', 'n/a')}")
                    st.markdown(f"**Citations:** {item.get('cited_by_count', 0)}")
                    st.markdown(f"**Topical fit:** {'YES' if fit else 'NO'}")
                    st.markdown(f"**Scholarly quality:** {'YES' if quality else 'NO'}")
                with tab_abstract:
                    st.markdown(item.get("abstract") or "Abstract not available.")
                with tab_links:
                    if item.get("doi"):
                        st.markdown(f"DOI: `{item.get('doi')}`")
                    if item.get("url"):
                        st.markdown(f"OpenAlex: {item.get('url')}")
                    if not item.get("doi") and not item.get("url"):
                        st.caption("No external links available for this work.")
                st.markdown("---")
            st.markdown("#### 🧭 Agent trace")
            write_trace_steps(run.get("trace") or [])
