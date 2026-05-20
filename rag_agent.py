# -*- coding: utf-8 -*-

import hashlib
import json
import pickle
import re

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from config import (
    CHAT_MODEL_NAME,
    CHUNK_OVERLAP,
    CHUNK_OVERLAP_PARAGRAPHS,
    CHUNK_SIZE,
    DEEPSEEK_BASE_URL,
    EMBEDDING_MODEL_NAME,
    INDEX_DIR,
    INDEX_FILE,
    KEYWORD_SCORE_WEIGHT,
    RERANK_TOP_K,
    REWRITE_HISTORY_MESSAGES,
    SEARCH_SCORE_RATIO,
    SEARCH_TOP_K,
)
from document_loader import load_documents
from logger import get_logger

logger = get_logger(__name__)


def split_paragraphs(text):
    paragraphs = []
    current_lines = []
    paragraph_start = None
    position = 0

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        line_start = position
        line_end = line_start + len(line)
        position = line_end

        if not stripped:
            if current_lines:
                paragraph_text = " ".join(item["text"] for item in current_lines)
                paragraphs.append(
                    {
                        "text": paragraph_text,
                        "char_start": paragraph_start,
                        "char_end": current_lines[-1]["char_end"],
                    }
                )
                current_lines = []
                paragraph_start = None
            continue

        if paragraph_start is None:
            paragraph_start = line_start

        current_lines.append(
            {
                "text": stripped,
                "char_end": line_end,
            }
        )

    if current_lines:
        paragraph_text = " ".join(item["text"] for item in current_lines)
        paragraphs.append(
            {
                "text": paragraph_text,
                "char_start": paragraph_start,
                "char_end": current_lines[-1]["char_end"],
            }
        )

    return paragraphs


def split_long_paragraph(paragraph, chunk_size):
    chunks = []

    for start in range(0, len(paragraph["text"]), chunk_size):
        chunk_text_value = paragraph["text"][start : start + chunk_size]
        chunks.append(
            {
                "text": chunk_text_value,
                "char_start": paragraph["char_start"] + start,
                "char_end": paragraph["char_start"] + start + len(chunk_text_value),
            }
        )

    return chunks


def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap_paragraphs=CHUNK_OVERLAP_PARAGRAPHS,
):
    chunks = []
    current_paragraphs = []
    current_length = 0

    for paragraph in split_paragraphs(text):
        paragraph_text = paragraph["text"]

        if len(paragraph_text) > chunk_size:
            if current_paragraphs:
                chunks.append(
                    {
                        "text": "\n\n".join(item["text"] for item in current_paragraphs),
                        "char_start": current_paragraphs[0]["char_start"],
                        "char_end": current_paragraphs[-1]["char_end"],
                    }
                )
                current_paragraphs = []
                current_length = 0

            chunks.extend(split_long_paragraph(paragraph, chunk_size))
            continue

        next_length = current_length + len(paragraph_text)
        if current_paragraphs:
            next_length += 2

        if current_paragraphs and next_length > chunk_size:
            chunks.append(
                {
                    "text": "\n\n".join(item["text"] for item in current_paragraphs),
                    "char_start": current_paragraphs[0]["char_start"],
                    "char_end": current_paragraphs[-1]["char_end"],
                }
            )
            current_paragraphs = current_paragraphs[-overlap_paragraphs:]
            current_length = sum(len(item["text"]) for item in current_paragraphs)

        current_paragraphs.append(paragraph)
        current_length += len(paragraph_text)

    if current_paragraphs:
        chunks.append(
            {
                "text": "\n\n".join(item["text"] for item in current_paragraphs),
                "char_start": current_paragraphs[0]["char_start"],
                "char_end": current_paragraphs[-1]["char_end"],
            }
        )

    return chunks


def get_text_fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_chunks(documents):
    chunks = []

    for document in documents:
        for index, chunk in enumerate(chunk_text(document["text"])):
            text = chunk["text"]
            chunks.append(
                {
                    "source": document["path"],
                    "index": index,
                    "text": text,
                    "char_start": chunk["char_start"],
                    "char_end": chunk["char_end"],
                    "fingerprint": get_text_fingerprint(text),
                }
            )

    return chunks


def cosine_similarity(vector_a, vector_b):
    return float(
        np.dot(vector_a, vector_b)
        / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    )


def extract_search_terms(question):
    terms = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", question.lower())
    expanded_terms = set(terms)

    for term in terms:
        if re.fullmatch(r"[\u4e00-\u9fff]{3,}", term):
            for size in (2, 3, 4):
                for index in range(0, len(term) - size + 1):
                    expanded_terms.add(term[index : index + size])

    return expanded_terms


def keyword_score(question, chunk):
    terms = extract_search_terms(question)
    if not terms:
        return 0

    text = chunk["text"].lower()
    source = chunk["source"].lower()
    score = 0

    for term in terms:
        if term in text:
            score += 1

        if term in source:
            score += 2

    return score


def build_embedding_index(chunks, model):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    return chunks


def save_embedding_index(chunks, index_file=INDEX_FILE):
    index_file.parent.mkdir(parents=True, exist_ok=True)

    with index_file.open("wb") as file:
        pickle.dump(chunks, file)


def load_embedding_index(index_file=INDEX_FILE):
    if not index_file.exists():
        return None

    with index_file.open("rb") as file:
        return pickle.load(file)


def is_cache_valid(current_chunks, cached_chunks):
    if cached_chunks is None:
        return False

    if len(current_chunks) != len(cached_chunks):
        return False

    for current_chunk, cached_chunk in zip(current_chunks, cached_chunks):
        if current_chunk["source"] != cached_chunk.get("source"):
            return False

        if current_chunk["index"] != cached_chunk.get("index"):
            return False

        if current_chunk["fingerprint"] != cached_chunk.get("fingerprint"):
            return False

    return True


def semantic_search(question, chunks, model, top_k=SEARCH_TOP_K):
    question_embedding = model.encode(question, normalize_embeddings=True)
    scored_chunks = []

    for chunk in chunks:
        semantic_score = cosine_similarity(question_embedding, chunk["embedding"])
        keyword_match_score = keyword_score(question, chunk)
        final_score = semantic_score + keyword_match_score * KEYWORD_SCORE_WEIGHT
        scored_chunks.append(
            (
                final_score,
                semantic_score,
                keyword_match_score,
                chunk,
            )
        )

    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    if not scored_chunks:
        return []

    max_score = scored_chunks[0][0]
    min_score = max_score * SEARCH_SCORE_RATIO

    filtered_chunks = [
        (score, semantic_score, keyword_match_score, chunk)
        for score, semantic_score, keyword_match_score, chunk in scored_chunks
        if score >= min_score
    ]

    return [
        {
            "score": round(score, 4),
            "semantic_score": round(semantic_score, 4),
            "keyword_score": keyword_match_score,
            "chunk": chunk,
        }
        for score, semantic_score, keyword_match_score, chunk in filtered_chunks[:top_k]
    ]


def build_context(relevant_chunks):
    return "\n\n".join(
        f"Source: {chunk['source']}\nContent:\n{chunk['text']}"
        for chunk in relevant_chunks
    )


def get_unique_sources(relevant_chunks):
    sources = []

    for chunk in relevant_chunks:
        source = chunk["source"]
        if source not in sources:
            sources.append(source)

    return sources


def answer_question(client, question, relevant_chunks):
    if not relevant_chunks:
        return "\u6211\u6ca1\u6709\u5728\u672c\u5730\u6587\u6863\u91cc\u627e\u5230\u76f8\u5173\u5185\u5bb9\u3002"

    chunks = [result["chunk"] for result in relevant_chunks]
    context = build_context(chunks)

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a local document Q&A agent. "
                        "Answer in Chinese. "
                        "Use only the provided document content. "
                        "If the answer is not in the documents, say you cannot find it."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Document content:\n{context}\n\nQuestion:\n{question}",
                },
            ],
        )
    except Exception as error:
        logger.error("Answer generation failed: %s", error)
        return (
            "\u56de\u7b54\u751f\u6210\u5931\u8d25\uff0c\u8bf7\u7a0d\u540e\u91cd\u8bd5\u3002"
            f" Error: {error}"
        )

    return response.choices[0].message.content


def preview_for_rerank(text, max_length=500):
    text = " ".join(text.split())

    if len(text) <= max_length:
        return text

    return text[:max_length] + "..."


def parse_rerank_indexes(content, max_index):
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []

    if isinstance(data, dict):
        indexes = data.get("indexes", [])
    elif isinstance(data, list):
        indexes = data
    else:
        return []

    valid_indexes = []
    for index in indexes:
        if isinstance(index, int) and 1 <= index <= max_index:
            if index not in valid_indexes:
                valid_indexes.append(index)

    return valid_indexes


def fallback_rerank_matches(matches, top_k=RERANK_TOP_K):
    logger.warning("Using fallback rerank for %s matches", len(matches))
    fallback_matches = []

    for index, match in enumerate(matches[:top_k], start=1):
        match["rerank_rank"] = index
        match["rerank_failed"] = True
        fallback_matches.append(match)

    return fallback_matches


def rerank_matches(client, question, matches, top_k=RERANK_TOP_K):
    if len(matches) <= top_k:
        for index, match in enumerate(matches, start=1):
            match["rerank_rank"] = index
            match["rerank_failed"] = False
        return matches

    candidate_text = "\n\n".join(
        (
            f"[{index}]\n"
            f"Source: {match['chunk']['source']} #{match['chunk']['index']}\n"
            f"Text: {preview_for_rerank(match['chunk']['text'])}"
        )
        for index, match in enumerate(matches, start=1)
    )

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a reranker for a document Q&A system. "
                        "Choose the candidate chunks that are most useful for answering "
                        "the user's question. Return only JSON in this format: "
                        "{\"indexes\": [1, 2, 3]}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Candidate chunks:\n{candidate_text}\n\n"
                        f"Return the best {top_k} candidate indexes."
                    ),
                },
            ],
        )
    except Exception as error:
        logger.warning("Rerank failed: %s", error)
        return fallback_rerank_matches(matches, top_k)

    selected_indexes = parse_rerank_indexes(
        response.choices[0].message.content,
        len(matches),
    )

    if not selected_indexes:
        return fallback_rerank_matches(matches, top_k)

    reranked_matches = []
    for rank, selected_index in enumerate(selected_indexes[:top_k], start=1):
        match = matches[selected_index - 1]
        match["rerank_rank"] = rank
        match["rerank_failed"] = False
        reranked_matches.append(match)

    return reranked_matches


def format_history_for_rewrite(history, max_messages=REWRITE_HISTORY_MESSAGES):
    if not history:
        return ""

    recent_messages = history[-max_messages:]
    lines = []

    for message in recent_messages:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)


def rewrite_question(client, question, history):
    history_text = format_history_for_rewrite(history)
    if not history_text:
        return question

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the user's latest question into a standalone "
                        "Chinese question for document retrieval. "
                        "Use the conversation history only to resolve references "
                        "such as it, this, that, the above, or the previous topic. "
                        "Do not answer the question. Return only the rewritten question."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Conversation history:\n{history_text}\n\n"
                        f"Latest question:\n{question}"
                    ),
                },
            ],
        )
    except Exception as error:
        logger.warning("Question rewrite failed: %s", error)
        return question

    rewritten = response.choices[0].message.content.strip()
    return rewritten or question


class RagAgent:
    def __init__(self, api_key, rebuild=False, extra_dirs=None, index_file=None):
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.documents = []
        self.document_errors = []
        self.chunks = []
        self.rebuild = rebuild
        self.extra_dirs = extra_dirs or []
        self.index_file = index_file or INDEX_FILE


    def load(self):
        logger.info("Loading documents")
        self.documents, self.document_errors = load_documents(
            extra_dirs=self.extra_dirs
        )
        self.chunks = build_chunks(self.documents)
        logger.info(
            "Loaded %s documents, %s document errors, %s chunks",
            len(self.documents),
            len(self.document_errors),
            len(self.chunks),
        )

        cached_chunks = None if self.rebuild else load_embedding_index(self.index_file)

        if is_cache_valid(self.chunks, cached_chunks):
            self.chunks = cached_chunks
            logger.info("Loaded embedding index from cache")
            return "cache"

        logger.info("Building embedding index")
        self.chunks = build_embedding_index(self.chunks, self.embedding_model)
        save_embedding_index(self.chunks, self.index_file)
        logger.info("Saved rebuilt embedding index")
        return "rebuilt"

    def ask(self, question, history=None):
        logger.info("Received question: %s", question)
        standalone_question = rewrite_question(self.client, question, history)
        logger.info("Standalone question: %s", standalone_question)
        matches = semantic_search(
            standalone_question,
            self.chunks,
            self.embedding_model,
        )
        logger.info("Hybrid search returned %s matches", len(matches))
        matches = rerank_matches(self.client, standalone_question, matches)
        logger.info("Rerank returned %s matches", len(matches))
        answer = answer_question(self.client, standalone_question, matches)
        chunks = [result["chunk"] for result in matches]

        return {
            "answer": answer,
            "standalone_question": standalone_question,
            "sources": get_unique_sources(chunks),
            "matches": matches,
        }
