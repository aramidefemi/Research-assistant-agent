from __future__ import annotations

import streamlit as st


def render_top_nav(*, on_admin_page: bool) -> None:
    """Render resilient top navigation across Streamlit versions/deploys."""
    col1, col2 = st.columns(2)
    with col1:
        if on_admin_page:
            if st.button("Go to Research workspace"):
                st.switch_page("app.py")
        else:
            st.caption("You are in Research workspace")
    with col2:
        if on_admin_page:
            st.caption("You are in Admin usage stats")
        else:
            if st.button("Go to Admin usage stats"):
                st.switch_page("pages/Admin_Usage_Stats.py")
