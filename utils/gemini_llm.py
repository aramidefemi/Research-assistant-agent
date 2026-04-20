"""Centralized LLM invocation with env-driven provider selection."""
from __future__ import annotations

import json
import os
import threading
import time
from urllib import error as urlerror
from urllib import request as urlrequest

from dotenv import load_dotenv

_ENV_LOADED = False
_RATE_LOCK = threading.Lock()
_LAST_CALL_AT = 0.0


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


def _has_openrouter() -> bool:
    return bool(_config_str("OPENROUTER_API_KEY") and _config_str("OPENROUTER_MODEL"))


def _config_float(key: str, default: float) -> float:
    raw = _config_str(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _config_int(key: str, default: int) -> int:
    raw = _config_str(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _config_bool(key: str, default: bool) -> bool:
    raw = _config_str(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _wait_for_rate_slot(min_interval_s: float) -> None:
    global _LAST_CALL_AT
    with _RATE_LOCK:
        now = time.time()
        wait_s = (_LAST_CALL_AT + min_interval_s) - now
        if wait_s > 0:
            time.sleep(wait_s)
        _LAST_CALL_AT = time.time()


def _invoke_gemini(prompt: str) -> str:
    import google.generativeai as genai

    keys = _key_chain()
    if not keys:
        raise RuntimeError(
            "Set GEMINI_API_KEY (and optionally GEMINI_API_KEY_ALT) in .env or .streamlit/secrets.toml"
        )
    model_name = _config_str("GEMINI_MODEL") or "gemini-2.0-flash"
    min_interval_s = _config_float("GEMINI_MIN_INTERVAL_SECONDS", 4.0)
    retry_count = _config_int("GEMINI_RETRY_COUNT", 3)
    backoff_s = _config_float("GEMINI_RETRY_BACKOFF_SECONDS", 3.0)

    last: BaseException | None = None
    for i, api_key in enumerate(keys):
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        for attempt in range(retry_count):
            try:
                _wait_for_rate_slot(min_interval_s)
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
                if not _is_quota_exceeded(e):
                    raise
                if attempt < retry_count - 1:
                    time.sleep(backoff_s * (2**attempt))
                    continue
                if i == len(keys) - 1:
                    raise
    assert last is not None
    raise last


def _http_error_message(err: urlerror.HTTPError) -> str:
    payload = ""
    try:
        payload = err.read().decode("utf-8", errors="replace")
    except Exception:
        payload = ""
    if payload:
        return f"HTTP {err.code}: {payload}"
    return f"HTTP {err.code}: {err.reason}"


def _invoke_openrouter(prompt: str) -> str:
    api_key = _config_str("OPENROUTER_API_KEY")
    model = _config_str("OPENROUTER_MODEL")
    if not api_key or not model:
        raise RuntimeError("Set OPENROUTER_API_KEY and OPENROUTER_MODEL to enable OpenRouter fallback.")

    endpoint = _config_str("OPENROUTER_API_URL") or "https://openrouter.ai/api/v1/chat/completions"
    timeout_s = _config_float("OPENROUTER_TIMEOUT_SECONDS", 60.0)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urlrequest.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": _config_str("OPENROUTER_SITE_URL") or "http://localhost",
            "X-Title": _config_str("OPENROUTER_APP_NAME") or "research-assistant",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urlerror.HTTPError as err:
        raise RuntimeError(f"OpenRouter call failed: {_http_error_message(err)}") from err
    except urlerror.URLError as err:
        raise RuntimeError(f"OpenRouter network error: {err.reason}") from err

    try:
        text = (body["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as err:
        raise RuntimeError("OpenRouter returned unexpected response format.") from err
    if not text:
        raise RuntimeError("OpenRouter returned an empty response.")
    return text


def _invoke_openai_compatible(prompt: str) -> str:
    try:
        from openai import OpenAI
    except Exception as err:
        raise RuntimeError("Install `openai` package to use OPENAI provider.") from err

    api_key = _config_str("OPENAI_API_KEY")
    model = _config_str("OPENAI_MODEL")
    if not api_key or not model:
        raise RuntimeError("Set OPENAI_API_KEY and OPENAI_MODEL to enable OPENAI provider.")

    base_url = _config_str("OPENAI_BASE_URL")
    temperature = _config_float("OPENAI_TEMPERATURE", 1.0)
    top_p = _config_float("OPENAI_TOP_P", 0.95)
    max_tokens = _config_int("OPENAI_MAX_TOKENS", 4096)
    timeout_s = _config_float("OPENAI_TIMEOUT_SECONDS", 60.0)
    enable_thinking = _config_bool("OPENAI_ENABLE_THINKING", False)
    reasoning_budget = _config_int("OPENAI_REASONING_BUDGET", max_tokens)

    client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=timeout_s)

    extra_body = None
    if enable_thinking:
        extra_body = {
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_budget": reasoning_budget,
        }

    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if extra_body is not None:
        kwargs["extra_body"] = extra_body

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as err:
        raise RuntimeError(f"OPENAI provider call failed: {err}") from err

    try:
        message = resp.choices[0].message
        text = (message.content or "").strip()
    except (AttributeError, IndexError, TypeError) as err:
        raise RuntimeError("OPENAI provider returned unexpected response format.") from err
    if not text:
        raise RuntimeError("OPENAI provider returned an empty response.")
    return text


def invoke_gemini_prompt(prompt: str) -> str:
    provider = (_config_str("LLM_PROVIDER") or "").strip().lower()
    if provider in {"openai", "openai_compatible", "nvidia"}:
        return _invoke_openai_compatible(prompt)
    if provider == "openrouter":
        return _invoke_openrouter(prompt)
    if provider == "gemini":
        return _invoke_gemini(prompt)

    openrouter_error: Exception | None = None
    if _has_openrouter():
        try:
            return _invoke_openrouter(prompt)
        except Exception as e:
            openrouter_error = e
    try:
        return _invoke_gemini(prompt)
    except Exception as e:
        if _is_quota_exceeded(e) and _has_openrouter():
            return _invoke_openrouter(prompt)
        if openrouter_error is not None:
            raise RuntimeError(
                f"OpenRouter failed first ({openrouter_error}); Gemini fallback also failed ({e})."
            ) from e
        raise
