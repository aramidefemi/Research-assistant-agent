from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from utils.pdf_reader import extract_text_from_pdf
from graph.pipeline import pipeline

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

    st.markdown("---")

    if st.session_state.results:
        total = len(st.session_state.results)
        fits = sum(1 for r in st.session_state.results if r.get("fit"))
        st.markdown(f"**Papers processed:** {total}")
        st.markdown(f"**Relevant:** {fits} / {total}")
        st.markdown("---")
        if st.button("🗑 Clear all results"):
            st.session_state.results = []
            st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("## Paper Evaluator")
st.markdown("Upload one or more journal PDFs. The agent will summarise each paper and score its relevance to your research focus.")

if not st.session_state.research_focus:
    st.info("👈 Set your research focus in the sidebar before uploading papers.")

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
                    }

                    result = pipeline.invoke(initial_state)
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

            tab1, tab2, tab3 = st.tabs(["📝 Summary", "🔑 Key Findings", "🔧 Methodology"])

            with tab1:
                st.markdown(result.get("summary", "Not available."))

            with tab2:
                st.markdown(result.get("key_findings", "Not available."))

            with tab3:
                st.markdown(result.get("methodology", "Not available."))
