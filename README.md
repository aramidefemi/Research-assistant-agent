# Research Assistant

An AI agent built with LangGraph + Streamlit that evaluates academic papers against your research focus.

## What it does

1. Upload one or more PDF papers
2. Agent extracts text → summarises → scores relevance to your defined research focus
3. Results show: fit verdict, relevance score, summary, key findings, methodology

## Project structure

```
research-assistant/
├── app.py                   # Streamlit UI
├── graph/
│   ├── state.py             # LangGraph state schema
│   ├── nodes.py             # extract / summarise / evaluate nodes
│   └── pipeline.py          # graph assembly
├── utils/
│   ├── pdf_reader.py        # PDF text extraction
│   └── prompts.py           # LLM prompt templates
├── .streamlit/
│   └── secrets.toml         # API keys (never commit this)
├── requirements.txt
└── .gitignore
```

## Local setup

```bash
pip install -r requirements.txt
# Add your Anthropic API key to .streamlit/secrets.toml
streamlit run app.py
```

## Streamlit Cloud deployment

1. Push to a GitHub repo (secrets.toml is gitignored)
2. Connect repo on share.streamlit.io
3. Go to App Settings → Secrets and add:
   ```
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
4. Deploy
