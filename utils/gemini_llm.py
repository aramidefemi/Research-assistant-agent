"""Call Gemini with GEMINI_API_KEY, fall back to GEMINI_API_KEY_ALT on quota / rate limits."""
from __future__ import annotations

import os

from dotenv import load_dotenv

_ENV_LOADED = False


def _ensure_dotenv() -> None:
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv()
        _ENV_LOADED = True


def _config_str(key: str) -> str | None:
    _ensure_dotenv()
    v = (os.environ.get(key) or "").strip()
    if v:
        return v
    try:
        import streamlit as st

        if key not in st.secrets:
            return None
        x = st.secrets[key]
        return x.strip() if isinstance(x, str) and x.strip() else None
    except Exception:
        return None


def _is_quota_exceeded(exc: BaseException) -> bool:
    try:
        from google.api_core import exceptions as gexc

        if isinstance(exc, (gexc.ResourceExhausted, gexc.TooManyRequests)):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    t = type(exc).__name__.lower()
    if "429" in msg or "resource_exhausted" in msg.replace(" ", "_"):
        return True
    if "quota" in msg and any(s in msg for s in ("exceed", "exceeded", "limit")):
        return True
    if "ratelimit" in t or "rate_limit" in msg or "too many requests" in msg:
        return True
    if "usage limit" in msg:
        return True
    return False


def _key_chain() -> list[str]:
    keys = (_config_str("GEMINI_API_KEY"), _config_str("GEMINI_API_KEY_ALT"))
    return [k for k in keys if k]


def invoke_gemini_prompt(prompt: str) -> str:
    import google.generativeai as genai

    keys = _key_chain()
    if not keys:
        raise RuntimeError(
            "Set GEMINI_API_KEY (and optionally GEMINI_API_KEY_ALT) in .env or .streamlit/secrets.toml"
        )
    model_name = _config_str("GEMINI_MODEL") or "gemini-2.0-flash"

    last: BaseException | None = None
    for i, api_key in enumerate(keys):
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        try:
            resp = model.generate_content(prompt)
            try:
                text = (resp.text or "").strip()
            except ValueError as ve:
                raise RuntimeError("Gemini returned no usable text (blocked or empty).") from ve
            if not text:
                raise RuntimeError("Gemini returned an empty response.")
            return text
        except Exception as e:
            last = e
            if not _is_quota_exceeded(e) or i == len(keys) - 1:
                raise
    assert last is not None
    raise last
