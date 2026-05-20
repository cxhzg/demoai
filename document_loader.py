# -*- coding: utf-8 -*-

from docx import Document
from pypdf import PdfReader

from config import DOCS_DIR, SUPPORTED_EXTENSIONS
from logger import get_logger


logger = get_logger(__name__)


def read_text_file(path):
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path):
    reader = PdfReader(path)
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return "\n\n".join(pages)


def read_docx_file(path):
    document = Document(path)
    paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def read_document_file(path):
    suffix = path.suffix.lower()

    if suffix == ".docx":
        return read_docx_file(path)

    if suffix == ".pdf":
        return read_pdf_file(path)

    return read_text_file(path)


def load_documents(extra_dirs=None):
    documents = []
    errors = []
    document_dirs = [DOCS_DIR]

    if extra_dirs:
        document_dirs.extend(extra_dirs)

    for document_dir in document_dirs:
        if not document_dir.exists():
            continue

        for path in document_dir.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            try:
                text = read_document_file(path)
            except Exception as error:
                logger.warning("Failed to read document %s: %s", path, error)
                errors.append(
                    {
                        "path": str(path),
                        "error": str(error),
                    }
                )
                continue

            if text.strip():
                documents.append({"path": str(path), "text": text})

    return documents, errors
