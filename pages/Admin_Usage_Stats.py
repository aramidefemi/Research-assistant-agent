from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from utils.trace_store import fetch_usage_stats


def _pct(part: int, total: int) -> str:
    if total <= 0:
        return "0%"
    return f"{round((part / total) * 100)}%"


st.set_page_config(
    page_title="Admin usage stats",
    page_icon=":material/admin_panel_settings:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.page_link("app.py", label="Research workspace", icon=":material/home:")
st.page_link("pages/Admin_Usage_Stats.py", label="Admin usage stats", icon=":material/admin_panel_settings:")

st.title("Admin usage stats")
st.caption("Small operational view of persisted pipeline usage.")

stats = fetch_usage_stats(limit_recent=15)
if not stats.get("available"):
    st.warning("Usage stats unavailable. Configure MONGODB_URI to enable persisted metrics.")
    st.stop()

total = int(stats.get("total_runs") or 0)
errors = int(stats.get("error_runs") or 0)
fits = int(stats.get("fit_runs") or 0)
llm = int(stats.get("llm_runs") or 0)
fallbacks = int(stats.get("fallback_runs") or 0)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total runs", total)
c2.metric("Errors", f"{errors} ({_pct(errors, total)})")
c3.metric("Fit decisions", f"{fits} ({_pct(fits, total)})")
c4.metric("LLM-used runs", f"{llm} ({_pct(llm, total)})")
c5.metric("Fallback runs", f"{fallbacks} ({_pct(fallbacks, total)})")

recent = list(stats.get("recent_runs") or [])
st.subheader("Recent runs")
if not recent:
    st.info("No persisted runs yet.")
else:
    rows = [
        {
            "stored_at": r.get("stored_at") or "",
            "filename": r.get("filename") or "",
            "fit": bool(r.get("fit")),
            "llm_used": bool(r.get("llm_used")),
            "error": (r.get("error") or "")[:120],
            "research_focus": (r.get("research_focus") or "")[:90],
        }
        for r in recent
    ]
    st.dataframe(rows, use_container_width=True)
