# Research Assistant

An AI assistant built with **LangGraph** and **Streamlit**. It supports:
- **PDF evaluation flow**: evaluate academic PDFs against your research focus using a three-step graph (extract → summarise → evaluate).
- **Topic discovery flow**: take only a topic, search journal papers, and loop search + evaluation until it qualifies up to 2 strong works.

### Agent architecture (preview)

PDF text is validated, then **summarise** and **evaluate** call Gemini in sequence; state and per-node traces live on `PaperState`. Full detail: **[docs/agent-architecture.md](docs/agent-architecture.md)**.

```
PDF → extract → summarise → evaluate → END
Topic → discovery_init → discovery_search → discovery_evaluate ↺ until 2 qualified or max rounds
```

## What it does

1. Set your research focus in the sidebar.
2. Choose either:
   - Upload PDFs for direct paper evaluation.
   - Enter only a topic for journal discovery.
3. Topic discovery loops over external journal search and your evaluator until at least 2 qualified papers are found or max rounds are reached.
4. Results show score, fit, reasons, and an **Agent trace** with per-node contributions.

## Orchestration and tracing

The LangGraph pipeline runs **extract**, **summarise**, and **evaluate** as separate nodes. Each step appends to an in-memory trace shown in the UI.

If **`MONGODB_URI`** is set (`.env` or Streamlit secrets), each completed run is stored in MongoDB (`research_assistant.pipeline_traces`). When persistence succeeds, the trace tab shows a **stored run id** so you can prove multi-step execution beyond the UI.

## Project structure

```
research-assistant/
├── docs/
│   └── agent-architecture.md  # LangGraph nodes, state, tracing
├── app.py                   # Streamlit UI
├── graph/
│   ├── state.py             # LangGraph state (includes optional trace)
│   ├── trace.py             # trace step helpers
│   ├── nodes.py             # extract / summarise / evaluate nodes
│   └── pipeline.py          # graph wiring
├── utils/
│   ├── pdf_reader.py        # PDF text extraction
│   ├── prompts.py           # LLM prompts
│   ├── gemini_llm.py        # OpenRouter-first LLM adapter + Gemini fallback
│   └── trace_store.py       # optional MongoDB persistence
├── .streamlit/
│   └── secrets.toml         # local secrets (gitignored — do not commit)
├── .env.example             # env var template
├── requirements.txt
└── .gitignore
```

## Local setup

```bash
pip install -r requirements.txt
```

Configure credentials using either **environment variables** (e.g. copy `.env.example` to `.env`) or **`.streamlit/secrets.toml`** for Streamlit-only apps:

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Required — primary Google AI key |
| `GEMINI_MODEL` | Optional — defaults to `gemini-2.0-flash` |
| `GEMINI_API_KEY_ALT` | Optional — used if the primary key hits quota |
| `OPENROUTER_API_KEY` | Recommended — enables OpenRouter as primary provider |
| `OPENROUTER_MODEL` | Recommended — OpenRouter model slug used as primary provider |
| `OPENROUTER_API_URL` | Optional — defaults to `https://openrouter.ai/api/v1/chat/completions` |
| `OPENROUTER_SITE_URL` | Optional — sent as `HTTP-Referer` header for OpenRouter analytics |
| `OPENROUTER_APP_NAME` | Optional — sent as `X-Title` header |
| `OPENROUTER_TIMEOUT_SECONDS` | Optional — request timeout, defaults to `60` |
| `MONGODB_URI` | Optional — persists pipeline traces for demos / auditing |

Run:

```bash
streamlit run app.py
```

## Streamlit Cloud deployment

1. Push to a GitHub repo (keep secrets out of the repo).
2. Connect the repo on [share.streamlit.io](https://share.streamlit.io).
3. App Settings → Secrets — add at least:

   ```toml
   GEMINI_API_KEY = "..."
   GEMINI_MODEL = "gemini-2.0-flash"
   GEMINI_API_KEY_ALT = ""
   OPENROUTER_API_KEY = ""
   OPENROUTER_MODEL = ""
   OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
   OPENROUTER_SITE_URL = "http://localhost"
   OPENROUTER_APP_NAME = "research-assistant"
   OPENROUTER_TIMEOUT_SECONDS = "60"
   MONGODB_URI = ""
   ```

4. Deploy.
