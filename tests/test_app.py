import shutil
from pathlib import Path

import pytest

import app


@pytest.fixture
def local_tmp_dir():
    path = Path(".test_tmp")
    if path.exists():
        shutil.rmtree(path)

    path.mkdir()

    try:
        yield path
    finally:
        if path.exists():
            shutil.rmtree(path)


def test_get_available_upload_path_returns_original_when_available(
    local_tmp_dir,
):
    result = app.get_available_upload_path("report.pdf", local_tmp_dir)

    assert result == local_tmp_dir / "report.pdf"


def test_get_available_upload_path_avoids_overwrite(local_tmp_dir):
    (local_tmp_dir / "report.pdf").write_text("existing", encoding="utf-8")
    (local_tmp_dir / "report_1.pdf").write_text("existing", encoding="utf-8")
    result = app.get_available_upload_path("report.pdf", local_tmp_dir)

    assert result == local_tmp_dir / "report_2.pdf"


def test_clear_session_documents_removes_uploads_and_index(
    local_tmp_dir,
    monkeypatch,
):
    index_dir = local_tmp_dir / ".index"
    upload_dir = local_tmp_dir / ".uploads" / "session-1"
    session_index_dir = index_dir / "session-1"

    monkeypatch.setattr(app, "INDEX_DIR", index_dir)

    upload_dir.mkdir(parents=True)
    session_index_dir.mkdir(parents=True)
    (upload_dir / "report.pdf").write_text("uploaded", encoding="utf-8")
    (session_index_dir / "embeddings.pkl").write_text("cache", encoding="utf-8")

    app.clear_session_documents(upload_dir, "session-1")

    assert upload_dir.exists()
    assert list(upload_dir.iterdir()) == []
    assert not session_index_dir.exists()


def test_has_supported_documents_returns_false_for_empty_directory(local_tmp_dir):
    assert app.has_supported_documents(local_tmp_dir) is False


def test_has_supported_documents_returns_true_for_supported_file(local_tmp_dir):
    (local_tmp_dir / "report.md").write_text("hello", encoding="utf-8")

    assert app.has_supported_documents(local_tmp_dir) is True
