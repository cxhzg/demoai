# -*- coding: utf-8 -*-

import os
import shutil
import time
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from config import INDEX_DIR, SUPPORTED_EXTENSIONS, SUPPORTED_UPLOAD_TYPES
from rag_agent import RagAgent

UPLOADS_DIR = Path(".uploads")

st.set_page_config(
    page_title="Local Doc Agent",
    page_icon=":books:",
    layout="wide",
)


def get_agent(api_key, session_upload_dir_text=None, session_id=None):
    extra_dirs = []

    if session_upload_dir_text is not None:
        extra_dirs.append(Path(session_upload_dir_text))

    index_file = None

    if session_id is not None:
        index_file = get_session_index_file(session_id)

    agent = RagAgent(
        api_key=api_key,
        extra_dirs=extra_dirs,
        index_file=index_file,
    )
    index_status = agent.load()
    return agent, index_status


def get_session_index_file(session_id):
    return INDEX_DIR / session_id / "embeddings.pkl"


def get_session_upload_dir():
    upload_dir = UPLOADS_DIR / st.session_state.session_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def has_supported_documents(directory):
    if not directory.exists():
        return False

    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return True

    return False


def clear_session_documents(session_upload_dir, session_id):
    session_index_dir = get_session_index_file(session_id).parent

    if session_upload_dir.exists():
        shutil.rmtree(session_upload_dir)

    if session_index_dir.exists():
        shutil.rmtree(session_index_dir)

    session_upload_dir.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file):
    upload_dir = get_session_upload_dir()
    target_path = get_available_upload_path(uploaded_file.name, upload_dir)

    with target_path.open("wb") as file:
        file.write(uploaded_file.getbuffer())

    return target_path


def get_available_upload_path(filename, directory):
    original_path = directory / filename
    if not original_path.exists():
        return original_path

    stem = original_path.stem
    suffix = original_path.suffix
    counter = 1

    while True:
        candidate_path = directory / f"{stem}_{counter}{suffix}"
        if not candidate_path.exists():
            return candidate_path

        counter += 1


def init_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def init_processed_uploads():
    if "processed_uploads" not in st.session_state:
        st.session_state.processed_uploads = set()


def init_upload_widget_key():
    if "upload_widget_key" not in st.session_state:
        st.session_state.upload_widget_key = str(uuid.uuid4())


def reset_upload_widget():
    st.session_state.upload_widget_key = str(uuid.uuid4())


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


def render_document_list(documents, session_upload_dir):
    project_documents = []
    uploaded_documents = []

    session_upload_dir_text = str(session_upload_dir)

    for document in documents:
        path = document["path"]

        if path.startswith(session_upload_dir_text):
            uploaded_documents.append(path)
        else:
            project_documents.append(path)

    st.subheader("Project documents")
    if project_documents:
        for path in project_documents:
            st.write(path)
    else:
        st.caption("No project documents.")

    st.subheader("Your uploaded documents")
    if uploaded_documents:
        for path in uploaded_documents:
            st.write(path)
    else:
        st.caption("No uploaded documents.")


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

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    init_chat_history()
    init_processed_uploads()
    init_upload_widget_key()

    st.title("Local Document Q&A Agent")

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        st.error("Please configure DEEPSEEK_API_KEY in the .env file.")
        return

    session_upload_dir = get_session_upload_dir()
    has_uploaded_documents = has_supported_documents(session_upload_dir)
    agent, index_status = get_agent(
        api_key,
        session_upload_dir_text=(
            str(session_upload_dir) if has_uploaded_documents else None
        ),
        session_id=st.session_state.session_id if has_uploaded_documents else None,
    )

    with st.sidebar:
        st.header("Documents")
        st.write(f"Index status: {index_status}")
        st.write(f"Documents: {len(agent.documents)}")
        st.write(f"Chunks: {len(agent.chunks)}")
        st.caption(f"Session:{st.session_state.session_id}")

        if agent.document_errors:
            st.warning(f"Failed documents: {len(agent.document_errors)}")
            with st.expander("Document read errors"):
                for error in agent.document_errors:
                    st.write(f"{error['path']}: {error['error']}")

        uploaded_file = st.file_uploader(
            "Upload document",
            type=SUPPORTED_UPLOAD_TYPES,
            key=st.session_state.upload_widget_key,
        )

        if uploaded_file is not None:
            upload_key = f"{uploaded_file.name}:{uploaded_file.size}"

            if upload_key not in st.session_state.processed_uploads:
                saved_path = save_uploaded_file(uploaded_file)
                st.session_state.processed_uploads.add(upload_key)
                st.success(f"Uploaded: {saved_path}")
                st.info("Click Rebuild index to include this document.")
            else:
                st.info("This file has already been uploaded.")
        
        if st.button("Rebuild index"):
            st.session_state.messages = []
            st.rerun()

        if st.button("Clear uploaded documents"):
            clear_session_documents(
                session_upload_dir,
                st.session_state.session_id,
            )
            st.session_state.messages = []
            st.session_state.processed_uploads = set()
            reset_upload_widget()
            st.success("Uploaded documents cleared.")
            st.rerun()

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        render_document_list(agent.documents, session_upload_dir)

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
