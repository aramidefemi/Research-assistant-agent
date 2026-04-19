# Research Assistant

An AI assistant built with **LangGraph** and **Streamlit**. It evaluates academic PDFs against your research focus using a **three-step graph** (extract → summarise → evaluate), not a single prompt.

## What it does

1. Set your research focus in the sidebar.
2. Upload one or more PDFs.
3. For each paper, the pipeline validates text, generates a structured summary (summary, findings, methodology), then scores relevance to your focus.
4. Results show fit verdict, score, narrative fields, and an **Agent trace** tab listing each graph node, what it contributed, and timings for Gemini calls.

## Orchestration and tracing

The LangGraph pipeline runs **extract**, **summarise**, and **evaluate** as separate nodes. Each step appends to an in-memory trace shown in the UI.

If **`MONGODB_URI`** is set (`.env` or Streamlit secrets), each completed run is stored in MongoDB (`research_assistant.pipeline_traces`). When persistence succeeds, the trace tab shows a **stored run id** so you can prove multi-step execution beyond the UI.

## Project structure

```
research-assistant/
├── app.py                   # Streamlit UI
├── graph/
│   ├── state.py             # LangGraph state (includes optional trace)
│   ├── trace.py             # trace step helpers
│   ├── nodes.py             # extract / summarise / evaluate nodes
│   └── pipeline.py          # graph wiring
├── utils/
│   ├── pdf_reader.py        # PDF text extraction
│   ├── prompts.py           # LLM prompts
│   ├── gemini_llm.py        # Gemini API (keys + fallback key)
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
   MONGODB_URI = ""
   ```

4. Deploy.
