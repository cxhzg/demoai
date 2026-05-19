import numpy as np

from rag_agent import (
    build_chunks,
    chunk_text,
    fallback_rerank_matches,
    keyword_score,
    parse_rerank_indexes,
    semantic_search,
)


def test_parse_rerank_indexes_filters_invalid_and_duplicate_indexes():
    content = '{"indexes": [2, 1, 99, 2, "bad"]}'

    result = parse_rerank_indexes(content, max_index=3)

    assert result == [2, 1]


def test_parse_rerank_indexes_accepts_plain_list():
    content = "[3, 1, 1, 4]"

    result = parse_rerank_indexes(content, max_index=3)

    assert result == [3, 1]


def test_parse_rerank_indexes_returns_empty_for_bad_json():
    result = parse_rerank_indexes("not json", max_index=3)

    assert result == []


def test_fallback_rerank_matches_marks_fallback_and_limits_results():
    matches = [
        {"score": 0.9, "chunk": {"source": "a.md", "index": 0}},
        {"score": 0.8, "chunk": {"source": "b.md", "index": 0}},
        {"score": 0.7, "chunk": {"source": "c.md", "index": 0}},
    ]

    result = fallback_rerank_matches(matches, top_k=2)

    assert len(result) == 2
    assert result[0]["rerank_rank"] == 1
    assert result[1]["rerank_rank"] == 2
    assert result[0]["rerank_failed"] is True
    assert result[1]["rerank_failed"] is True


def test_keyword_score_counts_text_and_source_matches():
    chunk = {
        "source": "docs/api.md",
        "text": "DEEPSEEK_API_KEY configuration guide",
    }

    result = keyword_score("DEEPSEEK_API_KEY api", chunk)

    assert result >= 3


def test_chunk_text_keeps_short_paragraphs_together():
    text = "first paragraph\n\nsecond paragraph"

    chunks = chunk_text(text, chunk_size=100)

    assert len(chunks) == 1
    assert chunks[0]["text"] == "first paragraph\n\nsecond paragraph"
    assert chunks[0]["char_start"] == 0
    assert chunks[0]["char_end"] == len(text)


def test_chunk_text_splits_long_paragraph_with_positions():
    text = "a" * 10

    chunks = chunk_text(text, chunk_size=4)

    assert [chunk["text"] for chunk in chunks] == ["aaaa", "aaaa", "aa"]
    assert [(chunk["char_start"], chunk["char_end"]) for chunk in chunks] == [
        (0, 4),
        (4, 8),
        (8, 10),
    ]


def test_build_chunks_adds_position_and_fingerprint():
    documents = [{"path": "docs/test.md", "text": "hello\n\nworld"}]

    chunks = build_chunks(documents)

    assert len(chunks) == 1
    assert chunks[0]["source"] == "docs/test.md"
    assert chunks[0]["char_start"] == 0
    assert chunks[0]["char_end"] == len("hello\n\nworld")
    assert chunks[0]["fingerprint"]


class FakeEmbeddingModel:
    def encode(self, text, normalize_embeddings=True):
        return np.array([1.0, 0.0])


def test_semantic_search_returns_hybrid_scores():
    chunks = [
        {
            "source": "docs/api.md",
            "index": 0,
            "text": "DEEPSEEK_API_KEY setup",
            "embedding": np.array([1.0, 0.0]),
            "char_start": 0,
            "char_end": 20,
        },
        {
            "source": "docs/other.md",
            "index": 0,
            "text": "unrelated text",
            "embedding": np.array([0.8, 0.2]),
            "char_start": 0,
            "char_end": 14,
        },
    ]

    result = semantic_search("DEEPSEEK_API_KEY", chunks, FakeEmbeddingModel())

    assert result[0]["chunk"]["source"] == "docs/api.md"
    assert "score" in result[0]
    assert "semantic_score" in result[0]
    assert "keyword_score" in result[0]
    assert result[0]["keyword_score"] > 0
