# -*- coding: utf-8 -*-

import argparse
import os

from dotenv import load_dotenv

from rag_agent import RagAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Local document Q&A agent")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the embedding index cache",
    )
    return parser.parse_args()


def preview_text(text, max_length=120):
    text = " ".join(text.split())

    if len(text) <= max_length:
        return text

    return text[:max_length] + "..."


def print_matched_chunks(search_results):
    if not search_results:
        return

    print("Matched chunks:")

    for index, result in enumerate(search_results, start=1):
        chunk = result["chunk"]
        score = result["score"]
        semantic_score = result.get("semantic_score", score)
        keyword_score = result.get("keyword_score", 0)
        rerank_rank = result.get("rerank_rank", index)
        rerank_label = "fallback" if result.get("rerank_failed") else rerank_rank

        print(
            f"{index}. rerank={rerank_label} score={score} semantic={semantic_score} "
            f"keyword={keyword_score} {chunk['source']} #{chunk['index']} "
            f"chars {chunk['char_start']}-{chunk['char_end']}"
        )
        print(f"   {preview_text(chunk['text'])}")

    print()


def print_answer(result):
    print(f"\nAgent:\n{result['answer']}\n")

    if result["sources"]:
        print("Sources:")
        for index, source in enumerate(result["sources"], start=1):
            print(f"{index}. {source}")
        print()

    print_matched_chunks(result["matches"])


def main():
    args = parse_args()
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Please configure DEEPSEEK_API_KEY in the .env file.")
        return

    print("Loading agent...")
    agent = RagAgent(api_key=api_key, rebuild=args.rebuild)

    if args.rebuild:
        print("Rebuild requested. Ignoring existing embedding cache.")

    index_status = agent.load()
    if index_status == "cache":
        print("Loading embedding index from cache...")
    else:
        print("Building embedding index...")

    print(f"Loaded {len(agent.documents)} documents and {len(agent.chunks)} chunks.")
    print("Ask a question. Type exit to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        if not question:
            continue

        result = agent.ask(question)
        print_answer(result)


if __name__ == "__main__":
    main()
