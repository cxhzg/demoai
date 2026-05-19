# -*- coding: utf-8 -*-

from pathlib import Path


DOCS_DIR = Path("docs")
INDEX_DIR = Path(".index")
INDEX_FILE = INDEX_DIR / "embeddings.pkl"

SUPPORTED_EXTENSIONS = {".docx", ".md", ".pdf", ".txt"}
SUPPORTED_UPLOAD_TYPES = ["docx", "md", "pdf", "txt"]

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CHAT_MODEL_NAME = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
CHUNK_OVERLAP_PARAGRAPHS = 1

SEARCH_TOP_K = 8
SEARCH_SCORE_RATIO = 0.75
KEYWORD_SCORE_WEIGHT = 0.05
RERANK_TOP_K = 5

REWRITE_HISTORY_MESSAGES = 6
