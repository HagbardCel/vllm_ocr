"""Storage, commit validation, and path safety tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bookextract.errors import ProcessingError
from bookextract.models import CommitManifest
from bookextract.storage import (
    RunStore,
    reject_symlink_components,
    validate_commit,
    validate_commit_relative_path,
)
from tests.conftest import make_inference_environment


def test_validate_commit_relative_path_rejects_traversal() -> None:
    with pytest.raises(ProcessingError, match="path traversal"):
        validate_commit_relative_path("../secret")
    with pytest.raises(ProcessingError, match="invalid relative path"):
        validate_commit_relative_path("/absolute")


def test_validate_commit_round_trip(tmp_path: Path) -> None:
    commit_dir = tmp_path / "page-0001"
    commit_dir.mkdir()
    content = b'{"page_index": 0}'
    rel = "page-assessment.json"
    (commit_dir / rel).write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    manifest = CommitManifest(page_index=0, files={rel: digest})
    (commit_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    loaded = validate_commit(commit_dir)
    assert loaded.page_index == 0
    assert loaded.files[rel] == digest


def test_validate_commit_detects_hash_mismatch(tmp_path: Path) -> None:
    commit_dir = tmp_path / "page-0001"
    commit_dir.mkdir()
    rel = "data.bin"
    (commit_dir / rel).write_bytes(b"actual")
    manifest = CommitManifest(page_index=0, files={rel: "0" * 64})
    (commit_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProcessingError, match="hash mismatch"):
        validate_commit(commit_dir)


def test_run_store_write_commit_advances_head(run_dir: Path) -> None:
    store = RunStore(run_dir)
    manifest = store.write_commit(
        1,
        {
            "page-assessment.json": b"{}\n",
            "page-context.json": b"{}\n",
            "interpretation.json": b"{}\n",
            "provenance.json": b"{}\n",
        },
    )
    assert manifest.page_index == 0
    assert store.read_head().committed_page_count == 1
    assert (run_dir / "commits" / "page-0001" / "manifest.json").is_file()


def test_run_store_rejects_non_contiguous_commit(run_dir: Path) -> None:
    store = RunStore(run_dir)
    files = {
        "page-assessment.json": b"{}\n",
        "page-context.json": b"{}\n",
        "interpretation.json": b"{}\n",
        "provenance.json": b"{}\n",
    }
    store.write_commit(1, files)
    with pytest.raises(ProcessingError, match="non-contiguous"):
        store.write_commit(3, files)


def test_inference_environment_write_once(run_dir: Path) -> None:
    store = RunStore(run_dir)
    env = make_inference_environment()
    store.write_inference_environment(env)
    with pytest.raises(ProcessingError, match="write-once"):
        store.write_inference_environment(env)


def test_reject_symlink_components(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(ProcessingError, match="symlink"):
        reject_symlink_components(link / "child")
