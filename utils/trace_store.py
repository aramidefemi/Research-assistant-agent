"""Optional MongoDB persistence for pipeline traces and usage events."""
from __future__ import annotations

import hashlib
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _client_id_from_ip(ip: str) -> str:
    salt = (os.environ.get("USAGE_HASH_SALT") or "").strip()
    raw = f"{salt}:{ip}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def resolve_client_meta() -> dict[str, str]:
    """Return best-effort client metadata with anonymized identifier."""
    ip = "unknown"
    user_agent = ""
    try:
        import streamlit as st

        headers = dict(getattr(getattr(st, "context", None), "headers", {}) or {})
        xff = str(headers.get("X-Forwarded-For") or headers.get("x-forwarded-for") or "").strip()
        if xff:
            ip = xff.split(",")[0].strip() or "unknown"
        elif headers.get("X-Real-Ip") or headers.get("x-real-ip"):
            ip = str(headers.get("X-Real-Ip") or headers.get("x-real-ip")).strip() or "unknown"
        user_agent = str(headers.get("User-Agent") or headers.get("user-agent") or "").strip()[:180]
    except Exception:
        pass
    return {
        "client_id": _client_id_from_ip(ip),
        "user_agent": user_agent,
    }


def log_usage_event(event_type: str, payload: dict[str, Any] | None = None) -> str | None:
    """Persist one usage/error event document."""
    uri = _mongo_uri()
    if not uri:
        return None
    try:
        from pymongo import MongoClient

        client_meta = resolve_client_meta()
        doc = {
            "event_type": (event_type or "").strip() or "unknown_event",
            "stored_at": _now_iso(),
            **client_meta,
            **(payload or {}),
        }
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        coll = client["research_assistant"]["usage_events"]
        result = coll.insert_one(doc)
        return str(result.inserted_id)
    except Exception:
        return None


def log_app_error(
    *,
    error: BaseException | str,
    context: str,
    filename: str = "",
    extra: dict[str, Any] | None = None,
) -> str | None:
    message = str(error).strip()[:600]
    payload = {
        "error_type": type(error).__name__ if not isinstance(error, str) else "Error",
        "error_message": message,
        "error_context": context[:120],
        "filename": filename[:180],
    }
    if extra:
        payload.update(extra)
    return log_usage_event("app_error", payload)


def log_llm_usage(
    *,
    provider: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    mode: str = "sync",
) -> str | None:
    p = max(0, int(prompt_tokens or 0))
    c = max(0, int(completion_tokens or 0))
    t = max(0, int(total_tokens or 0))
    if t == 0:
        t = p + c
    return log_usage_event(
        "llm_usage",
        {
            "provider": provider[:40],
            "model": model[:80],
            "mode": mode[:20],
            "prompt_tokens": p,
            "completion_tokens": c,
            "total_tokens": t,
        },
    )


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
        "unique_clients": 0,
        "error_events": 0,
        "recent_errors": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
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
        events = client["research_assistant"]["usage_events"]
        unique_clients = len([x for x in events.distinct("client_id") if x and x != _client_id_from_ip("unknown")])
        error_events = events.count_documents({"event_type": "app_error"})
        token_docs = events.find(
            {"event_type": "llm_usage"},
            {"_id": 0, "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 1},
        )
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        for doc in token_docs:
            prompt_tokens += int(doc.get("prompt_tokens") or 0)
            completion_tokens += int(doc.get("completion_tokens") or 0)
            total_tokens += int(doc.get("total_tokens") or 0)
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        recent_errors = list(
            events.find(
                {"event_type": "app_error"},
                {
                    "_id": 0,
                    "stored_at": 1,
                    "error_type": 1,
                    "error_message": 1,
                    "error_context": 1,
                    "filename": 1,
                    "client_id": 1,
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
            "unique_clients": int(unique_clients),
            "error_events": int(error_events),
            "recent_errors": recent_errors,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(total_tokens),
        }
    except Exception:
        return empty
