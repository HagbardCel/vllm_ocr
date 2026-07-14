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
    manifest = CommitManifest(manifest_format_version=1, page_index=0, files={rel: digest})
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
    manifest = CommitManifest(manifest_format_version=1, page_index=0, files={rel: "0" * 64})
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
    tmp_dir = store.commit_tmp_dir_for(3)
    final_dir = store.commit_dir_for(3)
    with pytest.raises(ProcessingError, match="non-contiguous"):
        store.write_commit(3, files)
    assert not tmp_dir.exists()
    assert not final_dir.exists()
    assert store.read_head().committed_page_count == 1


def test_write_commit_orphan_recovered_after_head_write_failure(run_dir: Path, monkeypatch) -> None:
    store = RunStore(run_dir)
    files = {
        "page-assessment.json": b"{}\n",
        "page-context.json": b"{}\n",
        "interpretation.json": b"{}\n",
        "provenance.json": b"{}\n",
    }

    original_write_head = store.write_head

    def failing_write_head(page_number: int) -> None:
        if page_number == 1:
            raise OSError("simulated head write failure")
        original_write_head(page_number)

    monkeypatch.setattr(store, "write_head", failing_write_head)
    with pytest.raises(OSError, match="simulated head write failure"):
        store.write_commit(1, files)

    final_dir = store.commit_dir_for(1)
    assert final_dir.exists()
    assert store.read_head().committed_page_count == 0

    monkeypatch.setattr(store, "write_head", original_write_head)
    state, committed = store.recover()
    assert committed == 0
    assert not final_dir.exists()
    assert state.processed_page_count == 0


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


def test_recover_rejects_missing_head_with_commits(run_dir: Path) -> None:
    store = RunStore(run_dir)
    files = {
        "page-assessment.json": b"{}\n",
        "page-context.json": b"{}\n",
        "interpretation.json": b"{}\n",
        "provenance.json": b"{}\n",
    }
    store.write_commit(1, files)
    commit_dir = run_dir / "commits" / "page-0001"
    assert commit_dir.is_dir()

    (run_dir / "head.json").unlink()
    with pytest.raises(ProcessingError, match="missing or invalid head.json"):
        store.recover()
    assert commit_dir.is_dir()


@pytest.mark.parametrize(
    "head_payload",
    [
        "{not json",
        '{"head_format_version": 99, "committed_page_count": 0}',
        '{"committed_page_count": 0}',
    ],
    ids=["invalid-json", "bad-version", "missing-version"],
)
def test_recover_rejects_malformed_head_with_commits(
    run_dir: Path,
    head_payload: str,
) -> None:
    store = RunStore(run_dir)
    files = {
        "page-assessment.json": b"{}\n",
        "page-context.json": b"{}\n",
        "interpretation.json": b"{}\n",
        "provenance.json": b"{}\n",
    }
    store.write_commit(1, files)
    commit_dir = run_dir / "commits" / "page-0001"
    assert commit_dir.is_dir()

    (run_dir / "head.json").write_text(head_payload, encoding="utf-8")
    with pytest.raises(ProcessingError, match="invalid"):
        store.recover()
    assert commit_dir.is_dir()
