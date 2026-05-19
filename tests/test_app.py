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
    monkeypatch,
):
    monkeypatch.setattr(app, "DOCS_DIR", local_tmp_dir)

    result = app.get_available_upload_path("report.pdf")

    assert result == local_tmp_dir / "report.pdf"


def test_get_available_upload_path_avoids_overwrite(local_tmp_dir, monkeypatch):
    monkeypatch.setattr(app, "DOCS_DIR", local_tmp_dir)
    (local_tmp_dir / "report.pdf").write_text("existing", encoding="utf-8")
    (local_tmp_dir / "report_1.pdf").write_text("existing", encoding="utf-8")

    result = app.get_available_upload_path("report.pdf")

    assert result == local_tmp_dir / "report_2.pdf"
