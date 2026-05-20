# Local Document Q&A Agent

A local RAG-based document Q&A agent built with Streamlit, local embeddings, and a DeepSeek-compatible chat API.

## Features

- Reads Markdown, TXT, PDF, and Word documents from `docs/`
- Uses local sentence-transformers embeddings for semantic search
- Supports hybrid search with semantic and keyword scores
- Supports reranking with fallback
- Caches embeddings in `.index/`
- Provides a Streamlit web UI
- Supports per-session document upload
- Isolates uploaded files and embedding indexes by Streamlit session
- Lets users clear their own uploaded documents and session index
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

1. Put public project documents into `docs/`.
2. Upload private or temporary documents in the web sidebar.
3. Rebuild the index from the web UI or with `py agent.py --rebuild`.
4. Ask questions in the web app or CLI.
5. Inspect the answer, sources, matched chunks, semantic score, keyword score, and rerank status.

Uploaded files are session-scoped in the web app:

```text
.uploads/<session_id>/
```

If a session has uploaded documents, its embedding index is also session-scoped:

```text
.index/<session_id>/embeddings.pkl
```

If a session has no uploaded documents, the app uses the shared project index:

```text
.index/embeddings.pkl
```

Use the web sidebar button `Clear uploaded documents` to delete the current session's uploaded files, session index, upload widget state, and chat history.

## Project Structure

```text
app.py              Streamlit UI, uploads, session state, and web workflow
agent.py            CLI entry point
rag_agent.py        Chunking, embeddings, hybrid search, reranking, and answers
document_loader.py  Markdown, TXT, PDF, and Word document loading
config.py           Runtime configuration
logger.py           Logging setup
tests/              Pytest tests
docs/               Project documents
```

## Web Upload Behavior

The web app separates project documents from user uploads:

```text
Project documents       docs/
Your uploaded documents .uploads/<session_id>/
```

This prevents one browser session's uploaded documents from being indexed through another session's cache. The app also tracks processed uploads in `st.session_state` so the same file is not saved repeatedly when Streamlit reruns the script.

Important notes:

- Uploaded files are temporary application files, not permanent storage.
- On Streamlit Cloud, uploaded files may disappear when the app restarts or redeploys.
- API calls made by visitors still use the API key configured for the deployed app.
- For public apps, consider adding authentication or a shared password before sharing widely.

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
- In Streamlit Cloud, set secrets with TOML syntax:

```toml
DEEPSEEK_API_KEY = "your_deepseek_api_key"
```

- Do not commit private documents unless the repository is private and you are comfortable with that.
- `.env`, `.index/`, `.uploads/`, `logs/`, and cache folders are ignored by `.gitignore`.
- Streamlit Cloud storage is not ideal for long-term uploaded document storage.
- First deployment may be slow because `sentence-transformers` downloads the embedding model.

## Limitations

- Scanned PDFs require OCR, which is not included yet.
- Large embedding models may load slowly on free cloud resources.
- Uploaded files on Streamlit Cloud may not be permanent across restarts.
- Session isolation is suitable for a learning demo, but it is not a full user account or permission system.
