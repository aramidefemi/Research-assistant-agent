"""Optional MongoDB persistence for pipeline traces (demo / judges)."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

_ENV_LOADED = False


def _ensure_dotenv() -> None:
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv()
        _ENV_LOADED = True


def _mongo_uri() -> str | None:
    _ensure_dotenv()
    uri = (os.environ.get("MONGODB_URI") or "").strip()
    if uri:
        return uri
    try:
        import streamlit as st

        if "MONGODB_URI" not in st.secrets:
            return None
        x = st.secrets["MONGODB_URI"]
        return x.strip() if isinstance(x, str) and x.strip() else None
    except Exception:
        return None


def persist_pipeline_run(payload: dict[str, Any]) -> str | None:
    """Insert one trace document; returns inserted_id hex or None if skipped/failed."""
    uri = _mongo_uri()
    if not uri:
        return None
    try:
        from pymongo import MongoClient

        doc = {
            **payload,
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        coll = client["research_assistant"]["pipeline_traces"]
        result = coll.insert_one(doc)
        return str(result.inserted_id)
    except Exception:
        return None
