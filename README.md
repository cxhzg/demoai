# Local Document Q&A Agent

A local RAG-based document Q&A agent built with Streamlit, local embeddings, and a DeepSeek-compatible chat API.

## Features

- Reads Markdown, TXT, PDF, and Word documents from `docs/`
- Uses local sentence-transformers embeddings for semantic search
- Supports hybrid search with semantic and keyword scores
- Supports reranking with fallback
- Caches embeddings in `.index/`
- Provides a Streamlit web UI
- Supports document upload
- Shows sources and matched chunks
- Supports chat history and follow-up question rewriting
- Includes logging and pytest-based local tests

## Install

```powershell
cd F:\demoai
py -m pip install -r requirements.txt
```

For a reproducible environment, use:

```powershell
py -m pip install -r requirements.lock.txt
```

## Configure

Create a `.env` file in the project root:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
```

Do not commit `.env` to GitHub.

## Run the CLI

```powershell
py agent.py
```

Rebuild the embedding index:

```powershell
py agent.py --rebuild
```

## Run the Web App

```powershell
py -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Typical Workflow

1. Put documents into `docs/`, or upload documents in the web sidebar.
2. Rebuild the index from the web UI or with `py agent.py --rebuild`.
3. Ask questions in the web app or CLI.
4. Inspect the answer, sources, matched chunks, semantic score, keyword score, and rerank status.

## Main Configuration

Most runtime settings live in `config.py`:

```python
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
SEARCH_TOP_K = 8
SEARCH_SCORE_RATIO = 0.75
KEYWORD_SCORE_WEIGHT = 0.05
RERANK_TOP_K = 5
REWRITE_HISTORY_MESSAGES = 6
```

After changing chunk or embedding settings, rebuild the index.

## Tests

```powershell
py -m pytest
```

## Deployment Notes

- Keep `DEEPSEEK_API_KEY` in Streamlit Cloud secrets, not in the repository.
- Do not commit private documents unless the repository is private and you are comfortable with that.
- `.env`, `.index/`, `logs/`, and cache folders are ignored by `.gitignore`.
- Streamlit Cloud storage is not ideal for long-term uploaded document storage.

## Limitations

- Scanned PDFs require OCR, which is not included yet.
- Large embedding models may load slowly on free cloud resources.
- Uploaded files on Streamlit Cloud may not be permanent across restarts.
