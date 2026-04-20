from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from utils.pdf_reader import extract_text_from_pdf
from utils.trace_store import persist_pipeline_run
from graph.pipeline import pipeline, discovery_pipeline


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
        st.markdown(f"**Step {i} — `{node}`**")
        st.markdown(step.get("contribution", ""))
        det = step.get("detail") or ""
        if det:
            st.caption(det)
        dm = step.get("duration_ms")
        if dm is not None:
            st.caption(f"⏱ {dm:.0f} ms")
        st.markdown("---")


def run_pipeline_stream(graph, initial_state: dict, live_placeholder, run_label: str) -> dict:
    """Run a graph with streaming; updates `live_placeholder` after each node completes."""
    result = initial_state
    with live_placeholder.container():
        st.markdown(f"#### 🧭 Live agent trace — `{run_label}`")
        st.caption("Starting pipeline…")
    for chunk in graph.stream(initial_state, stream_mode="values"):
        result = chunk
        with live_placeholder.container():
            st.markdown(f"#### 🧭 Live agent trace — `{run_label}`")
            write_trace_steps(chunk.get("trace") or [])
    return result


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
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

# ── Session state init ────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = []
if "research_focus" not in st.session_state:
    st.session_state.research_focus = ""
if "evaluation_depth" not in st.session_state:
    st.session_state.evaluation_depth = "full"
if "discovery_results" not in st.session_state:
    st.session_state.discovery_results = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🔬 Research Assistant")
    st.markdown("---")

    st.markdown('<p class="sidebar-header">Your Research Focus</p>', unsafe_allow_html=True)
    research_focus = st.text_area(
        label="Research focus",
        label_visibility="collapsed",
        placeholder="e.g. Machine learning approaches to credit risk scoring using XGBoost and SHAP explainability in low-data environments...",
        height=160,
        key="research_focus_input",
    )
    if research_focus:
        st.session_state.research_focus = research_focus

    st.markdown('<p class="sidebar-header">Evaluation</p>', unsafe_allow_html=True)
    st.radio(
        "Evaluation depth",
        ["full", "quick"],
        format_func=lambda x: (
            "Full — score, fit, and why (2 LLM steps)" if x == "full" else "Quick — score & fit only (1 step)"
        ),
        key="evaluation_depth",
    )

    st.markdown("---")

    if st.session_state.results:
        total = len(st.session_state.results)
        fits = sum(1 for r in st.session_state.results if r.get("fit"))
        st.markdown(f"**Papers processed:** {total}")
        st.markdown(f"**Relevant:** {fits} / {total}")
    if st.session_state.discovery_results:
        d_total = len(st.session_state.discovery_results)
        st.markdown(f"**Discovery runs:** {d_total}")
        st.markdown("---")
        if st.button("🗑 Clear all results"):
            st.session_state.results = []
            st.session_state.discovery_results = []
            st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("## Research Agent")
st.markdown("Start with a topic and let the agent discover + qualify top journal works. Optionally add your own PDFs.")

if not st.session_state.research_focus:
    st.info("👈 Set your research focus in the sidebar before running the agent.")

tab_discovery, tab_pdf = st.tabs(["🤖 Topic Agent (Recommended)", "📄 Add Your PDFs (Optional)"])

with tab_discovery:
    st.markdown("Give only a topic. The orchestrator loops search + evaluation until it finds 2 qualified works (or reaches max rounds).")
    topic = st.text_input("Topic", placeholder="e.g. Explainable AI in credit risk scoring")
    col_a, col_b = st.columns(2)
    max_rounds = col_a.number_input("Max rounds", min_value=1, max_value=12, value=5)
    batch_size = col_b.number_input("Candidates per round", min_value=3, max_value=20, value=8)

    if st.button("▶ Run topic agent", type="primary") and topic:
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
            "topic": topic.strip(),
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
            run_status.update(label="✅ Topic agent finished", state="complete", expanded=False)
        st.session_state.discovery_results.append(result)
        st.rerun()

with tab_pdf:
    st.markdown("Optional: upload one or more PDFs to evaluate your own selected papers.")
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
            if st.button(f"▶ Run agent on {len(new_files)} paper(s)"):
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
                        st.session_state.results.append({
                            "filename": uploaded_file.name,
                            "error": str(e),
                            "fit": False,
                        })

                    progress.progress((i + 1) / len(new_files))

                status.empty()
                progress.empty()
                st.success("✅ Done! Results below.")
                st.rerun()

    elif uploaded_files and not st.session_state.research_focus:
        st.warning("Please set your research focus in the sidebar first.")

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
                if item.get("doi"):
                    st.markdown(f"DOI: `{item.get('doi')}`")
                if item.get("url"):
                    st.markdown(f"OpenAlex: {item.get('url')}")
                st.markdown(f"**Score:** `{item.get('score', 0):.2f}`")
                st.markdown(f"**Why:** {item.get('reason', '')}")
                st.markdown("---")
            st.markdown("#### 🧭 Agent trace")
            write_trace_steps(run.get("trace") or [])
