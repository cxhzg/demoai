# -*- coding: utf-8 -*-

import os
import time

import streamlit as st
from dotenv import load_dotenv

from config import DOCS_DIR, SUPPORTED_UPLOAD_TYPES
from rag_agent import RagAgent


st.set_page_config(
    page_title="Local Doc Agent",
    page_icon=":books:",
    layout="wide",
)


@st.cache_resource
def get_agent(api_key, rebuild=False):
    agent = RagAgent(api_key=api_key, rebuild=rebuild)
    index_status = agent.load()
    return agent, index_status


def save_uploaded_file(uploaded_file):
    DOCS_DIR.mkdir(exist_ok=True)
    target_path = get_available_upload_path(uploaded_file.name)

    with target_path.open("wb") as file:
        file.write(uploaded_file.getbuffer())

    return target_path


def get_available_upload_path(filename):
    original_path = DOCS_DIR / filename
    if not original_path.exists():
        return original_path

    stem = original_path.stem
    suffix = original_path.suffix
    counter = 1

    while True:
        candidate_path = DOCS_DIR / f"{stem}_{counter}{suffix}"
        if not candidate_path.exists():
            return candidate_path

        counter += 1


def init_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_sources(sources):
    if not sources:
        st.write("No sources found.")
        return

    for index, source in enumerate(sources, start=1):
        st.write(f"{index}. {source}")


def render_matches(matches):
    if not matches:
        st.write("No matched chunks.")
        return

    for index, match in enumerate(matches, start=1):
        chunk = match["chunk"]
        score = match["score"]
        semantic_score = match.get("semantic_score", score)
        keyword_score = match.get("keyword_score", 0)
        rerank_rank = match.get("rerank_rank", index)
        rerank_label = "fallback" if match.get("rerank_failed") else rerank_rank

        with st.expander(
            (
                f"{index}. rerank={rerank_label} score={score} semantic={semantic_score} "
                f"keyword={keyword_score} {chunk['source']} #{chunk['index']} "
                f"chars {chunk['char_start']}-{chunk['char_end']}"
            )
        ):
            st.write(chunk["text"])


def render_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message["role"] == "assistant":
                elapsed_seconds = message.get("elapsed_seconds")
                if elapsed_seconds is not None:
                    st.caption(f"Answered in {elapsed_seconds:.2f}s")

                rewritten_question = message.get("standalone_question")
                if rewritten_question:
                    st.caption(f"Rewritten question: {rewritten_question}")

                with st.expander("Sources"):
                    render_sources(message.get("sources", []))

                with st.expander("Matched chunks"):
                    render_matches(message.get("matches", []))


def main():
    load_dotenv()
    init_chat_history()

    st.title("Local Document Q&A Agent")

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        st.error("Please configure DEEPSEEK_API_KEY in the .env file.")
        return

    agent, index_status = get_agent(api_key)

    with st.sidebar:
        st.header("Documents")
        st.write(f"Index status: {index_status}")
        st.write(f"Documents: {len(agent.documents)}")
        st.write(f"Chunks: {len(agent.chunks)}")

        if agent.document_errors:
            st.warning(f"Failed documents: {len(agent.document_errors)}")
            with st.expander("Document read errors"):
                for error in agent.document_errors:
                    st.write(f"{error['path']}: {error['error']}")

        uploaded_file = st.file_uploader(
            "Upload document",
            type=SUPPORTED_UPLOAD_TYPES,
        )

        if uploaded_file is not None:
            saved_path = save_uploaded_file(uploaded_file)
            st.success(f"Uploaded: {saved_path}")
            st.info("Click Rebuild index to include this document.")
        
        if st.button("Rebuild index"):
            get_agent.clear()
            st.session_state.messages = []
            agent, index_status = get_agent(api_key, rebuild=True)
            st.success("Index rebuilt.")
            st.rerun()

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        for document in agent.documents:
            st.write(document["path"])

    render_chat_history()

    question = st.chat_input("Ask a question")

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.write(question)

        start_time = time.perf_counter()
        with st.spinner("Thinking..."):
            result = agent.ask(question, history=st.session_state.messages[:-1])
        elapsed_seconds = time.perf_counter() - start_time

        assistant_message = {
            "role": "assistant",
            "content": result["answer"],
            "elapsed_seconds": elapsed_seconds,
            "standalone_question": result["standalone_question"],
            "sources": result["sources"],
            "matches": result["matches"],
        }
        st.session_state.messages.append(assistant_message)

        with st.chat_message("assistant"):
            st.write(result["answer"])
            st.caption(f"Answered in {elapsed_seconds:.2f}s")
            st.caption(f"Rewritten question: {result['standalone_question']}")

            with st.expander("Sources"):
                render_sources(result["sources"])

            with st.expander("Matched chunks"):
                render_matches(result["matches"])


if __name__ == "__main__":
    main()
