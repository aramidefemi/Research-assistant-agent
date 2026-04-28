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


def fetch_usage_stats(limit_recent: int = 10) -> dict[str, Any]:
    """Return aggregate usage stats from stored pipeline traces."""
    empty = {
        "available": False,
        "total_runs": 0,
        "error_runs": 0,
        "fit_runs": 0,
        "llm_runs": 0,
        "fallback_runs": 0,
        "recent_runs": [],
    }
    uri = _mongo_uri()
    if not uri:
        return empty
    try:
        from pymongo import MongoClient

        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        coll = client["research_assistant"]["pipeline_traces"]
        total_runs = coll.count_documents({})
        error_runs = coll.count_documents({"error": {"$nin": [None, ""]}})
        fit_runs = coll.count_documents({"fit": True})
        llm_runs = coll.count_documents({"llm_used": True})
        fallback_runs = coll.count_documents({"fallback_reason": {"$nin": [None, ""]}})
        recent_cursor = (
            coll.find(
                {},
                {
                    "_id": 0,
                    "filename": 1,
                    "research_focus": 1,
                    "fit": 1,
                    "error": 1,
                    "llm_used": 1,
                    "stored_at": 1,
                },
            )
            .sort("stored_at", -1)
            .limit(max(1, int(limit_recent)))
        )
        return {
            "available": True,
            "total_runs": int(total_runs),
            "error_runs": int(error_runs),
            "fit_runs": int(fit_runs),
            "llm_runs": int(llm_runs),
            "fallback_runs": int(fallback_runs),
            "recent_runs": list(recent_cursor),
        }
    except Exception:
        return empty
