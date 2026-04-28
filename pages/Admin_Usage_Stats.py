try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv() -> None:
        return None

load_dotenv()

import streamlit as st

try:
    from utils.nav import render_page_nav
except Exception:
    def render_page_nav() -> None:
        st.markdown("[Research workspace](/) · [Admin usage stats](/Admin_Usage_Stats)")
from utils.trace_store import fetch_usage_stats, log_usage_event


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

render_page_nav()
if "usage_open_logged_admin" not in st.session_state:
    st.session_state.usage_open_logged_admin = False
if not st.session_state.usage_open_logged_admin:
    log_usage_event("app_page_open", {"page": "admin_usage_stats"})
    st.session_state.usage_open_logged_admin = True

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
unique_clients = int(stats.get("unique_clients") or 0)
error_events = int(stats.get("error_events") or 0)
prompt_tokens = int(stats.get("prompt_tokens") or 0)
completion_tokens = int(stats.get("completion_tokens") or 0)
total_tokens = int(stats.get("total_tokens") or 0)

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Total runs", total)
c2.metric("Errors", f"{errors} ({_pct(errors, total)})")
c3.metric("Fit decisions", f"{fits} ({_pct(fits, total)})")
c4.metric("LLM-used runs", f"{llm} ({_pct(llm, total)})")
c5.metric("Fallback runs", f"{fallbacks} ({_pct(fallbacks, total)})")
c6.metric("Unique users (anon IP)", unique_clients)
c7.metric("App error events", error_events)

t1, t2, t3 = st.columns(3)
t1.metric("Prompt tokens", prompt_tokens)
t2.metric("Completion tokens", completion_tokens)
t3.metric("Total tokens", total_tokens)

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

recent_errors = list(stats.get("recent_errors") or [])
st.subheader("Recent app errors")
if not recent_errors:
    st.info("No logged app errors yet.")
else:
    error_rows = [
        {
            "stored_at": r.get("stored_at") or "",
            "error_type": r.get("error_type") or "",
            "error_context": r.get("error_context") or "",
            "filename": r.get("filename") or "",
            "client_id": r.get("client_id") or "",
            "error_message": (r.get("error_message") or "")[:180],
        }
        for r in recent_errors
    ]
    st.dataframe(error_rows, use_container_width=True)

with st.expander("Other useful ideas", expanded=False):
    st.markdown(
        "- Track daily active users (distinct `client_id` per day).\n"
        "- Track avg latency per pipeline run from trace durations.\n"
        "- Add error rate by context (`pdf_upload_process`, `chat_response`).\n"
        "- Add token usage trend (today vs last 7 days)."
    )
