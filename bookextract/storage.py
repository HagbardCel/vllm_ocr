"""Run directory persistence with atomic writes and recovery."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bookextract.config import (
    InferenceLocation,
    RunRecord,
    SourceLocation,
    write_json_atomic,
)
from bookextract.errors import ProcessingError
from bookextract.models import (
    BookState,
    CommitManifest,
    HeadState,
    InferenceEnvironment,
    StateCache,
)

PAGE_COMMIT_RE = re.compile(r"^page-(\d{4})$")
PAGE_TMP_RE = re.compile(r"^\.page-(\d{4})\.tmp$")
FAILURE_DIR_RE = re.compile(r"^failure-(\d{4})$")
RECOVERY_DIR_RE = re.compile(r"^recovery-(\d{4})$")
PREFLIGHT_DIR_RE = re.compile(r"^preflight-(\d{4})$")


def reject_symlink_components(path: Path) -> None:
    """Reject any symlink component in a path."""
    current = path
    parts: list[Path] = []
    while True:
        parts.append(current)
        if current == current.parent:
            break
        current = current.parent
    for part in reversed(parts):
        if part.is_symlink():
            raise ProcessingError(
                code="invalid-commit-artifact-path",
                message=f"symlink component rejected: {part}",
            )


def validate_commit_relative_path(rel_path: str) -> None:
    """Validate a manifest-relative path is safe and nested correctly."""
    if not rel_path or rel_path.startswith("/") or "\\" in rel_path:
        raise ProcessingError(
            code="invalid-commit-artifact-path",
            message=f"invalid relative path: {rel_path!r}",
        )
    parts = rel_path.split("/")
    if ".." in parts or "" in parts:
        raise ProcessingError(
            code="invalid-commit-artifact-path",
            message=f"path traversal rejected: {rel_path!r}",
        )
    for part in parts:
        if part in (".", ".."):
            raise ProcessingError(
                code="invalid-commit-artifact-path",
                message=f"invalid path component: {rel_path!r}",
            )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_commit(commit_dir: Path) -> CommitManifest:
    """Validate commit directory manifest matches on-disk regular files."""
    reject_symlink_components(commit_dir)
    if not commit_dir.is_dir():
        raise ProcessingError(
            code="invalid-run-layout",
            message=f"missing commit directory: {commit_dir}",
        )

    manifest_path = commit_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ProcessingError(
            code="invalid-run-layout",
            message=f"missing manifest: {manifest_path}",
        )

    manifest = CommitManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )

    on_disk: dict[str, str] = {}
    for entry in commit_dir.rglob("*"):
        if not entry.is_file() or entry.name == "manifest.json":
            continue
        if entry.is_symlink():
            raise ProcessingError(
                code="invalid-commit-artifact-path",
                message=f"symlink in commit: {entry}",
            )
        rel = entry.relative_to(commit_dir).as_posix()
        validate_commit_relative_path(rel)
        on_disk[rel] = _sha256_file(entry)

    if set(manifest.files.keys()) != set(on_disk.keys()):
        raise ProcessingError(
            code="invalid-run-layout",
            message=(
                f"manifest mismatch in {commit_dir.name}: "
                f"manifest={sorted(manifest.files)} disk={sorted(on_disk)}"
            ),
        )

    for rel, expected_hash in manifest.files.items():
        if on_disk[rel] != expected_hash:
            raise ProcessingError(
                code="invalid-run-layout",
                message=f"hash mismatch for {rel} in {commit_dir.name}",
            )

    return manifest


@dataclass
class FailureRecord:
    failure_directory: Path
    failure_number: int


class RunStore:
    """Atomic run-directory persistence."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir.resolve()
        reject_symlink_components(self.run_dir)

    def _path(self, *parts: str) -> Path:
        path = self.run_dir.joinpath(*parts)
        reject_symlink_components(path)
        return path

    def ensure_layout(self) -> None:
        for name in (
            "pages",
            "commits",
            "failures",
            "diagnostics",
            "recovery",
            ".output-build",
            "output",
        ):
            self._path(name).mkdir(parents=True, exist_ok=True)

    def atomic_write_bytes(self, path: Path, content: bytes) -> None:
        reject_symlink_components(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_bytes(content)
        os.replace(tmp, path)

    def atomic_write_json(self, path: Path, payload: dict[str, Any]) -> None:
        self.atomic_write_bytes(
            path,
            (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
        )

    def atomic_write_text(self, path: Path, text: str) -> None:
        self.atomic_write_bytes(path, text.encode("utf-8"))

    def read_head(self) -> HeadState:
        head_path = self._path("head.json")
        if not head_path.is_file() or head_path.is_symlink():
            raise ProcessingError(
                code="invalid-run-layout",
                message="missing or invalid head.json",
            )
        try:
            return HeadState.model_validate_json(head_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ProcessingError(
                code="invalid-run-layout",
                message="invalid head.json",
            ) from exc

    def write_head(self, committed_page_count: int) -> None:
        payload = HeadState(
            head_format_version=1,
            committed_page_count=committed_page_count,
        ).model_dump(mode="json")
        head_path = self._path("head.json")
        tmp = self._path("head.json.tmp")
        self.atomic_write_json(tmp, payload)
        os.replace(tmp, head_path)

    def read_state_cache(self) -> StateCache | None:
        state_path = self._path("state.json")
        if not state_path.is_file():
            return None
        try:
            return StateCache.model_validate_json(state_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def write_state_cache(self, state: BookState, committed_page_count: int) -> None:
        cache = StateCache(
            state_cache_version=1,
            committed_page_count=committed_page_count,
            state=state,
        )
        write_json_atomic(self._path("state.json"), cache.model_dump(mode="json"))

    def load_run_record(self) -> RunRecord:
        return RunRecord.model_validate_json(
            self._path("run.json").read_text(encoding="utf-8")
        )

    def load_source_location(self) -> SourceLocation:
        return SourceLocation.model_validate_json(
            self._path("source-location.json").read_text(encoding="utf-8")
        )

    def write_source_location(self, location: SourceLocation) -> None:
        write_json_atomic(
            self._path("source-location.json"),
            location.model_dump(mode="json"),
        )

    def load_inference_location(self) -> InferenceLocation:
        path = self._path("inference-location.json")
        if not path.is_file():
            raise ProcessingError(
                code="invalid-run-layout",
                message="missing inference-location.json",
            )
        return InferenceLocation.model_validate_json(path.read_text(encoding="utf-8"))

    def write_inference_location(self, location: InferenceLocation) -> None:
        write_json_atomic(
            self._path("inference-location.json"),
            location.model_dump(mode="json"),
        )

    def load_inference_environment(self) -> InferenceEnvironment | None:
        path = self._path("inference-environment.json")
        if not path.is_file():
            return None
        return InferenceEnvironment.model_validate_json(path.read_text(encoding="utf-8"))

    def write_inference_environment(self, environment: InferenceEnvironment) -> None:
        path = self._path("inference-environment.json")
        if path.exists():
            raise ProcessingError(
                code="inference-environment-drift",
                message="inference-environment.json already exists (write-once)",
            )
        tmp = self._path("inference-environment.json.tmp")
        self.atomic_write_json(tmp, environment.model_dump(mode="json"))
        os.replace(tmp, path)

    def commit_dir_for(self, page_number: int) -> Path:
        return self._path("commits", f"page-{page_number:04d}")

    def commit_tmp_dir_for(self, page_number: int) -> Path:
        return self._path("commits", f".page-{page_number:04d}.tmp")

    def write_commit(
        self,
        page_number: int,
        files: dict[str, bytes],
    ) -> CommitManifest:
        """Write a complete commit atomically and advance head."""
        page_index = page_number - 1
        tmp_dir = self.commit_tmp_dir_for(page_number)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        try:
            head = self.read_head()
            expected = head.committed_page_count + 1
            if page_number != expected:
                raise ProcessingError(
                    code="invalid-run-layout",
                    message=(
                        f"non-contiguous commit: expected page "
                        f"{expected:04d}, got {page_number:04d}"
                    ),
                )

            file_hashes: dict[str, str] = {}
            for rel_path, content in files.items():
                validate_commit_relative_path(rel_path)
                dest = tmp_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(content)
                file_hashes[rel_path] = _sha256_bytes(content)

            manifest = CommitManifest(
                manifest_format_version=1,
                page_index=page_index,
                files=file_hashes,
            )
            manifest_bytes = (
                json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False)
                + "\n"
            ).encode("utf-8")
            (tmp_dir / "manifest.json").write_bytes(manifest_bytes)

            validate_commit(tmp_dir)

            final_dir = self.commit_dir_for(page_number)
            if final_dir.exists():
                raise ProcessingError(
                    code="invalid-run-layout",
                    message=f"commit already exists: {final_dir.name}",
                )
            os.replace(tmp_dir, final_dir)
            self.write_head(page_number)
            return manifest
        except BaseException:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            raise

    def read_commit_file(self, page_number: int, rel_path: str) -> bytes:
        validate_commit_relative_path(rel_path)
        path = self.commit_dir_for(page_number) / rel_path
        reject_symlink_components(path)
        return path.read_bytes()

    def _next_recovery_dir(self) -> Path:
        recovery_root = self._path("recovery")
        recovery_root.mkdir(parents=True, exist_ok=True)
        existing = [
            int(RECOVERY_DIR_RE.match(name).group(1))  # type: ignore[union-attr]
            for name in os.listdir(recovery_root)
            if RECOVERY_DIR_RE.match(name)
        ]
        next_num = (max(existing) + 1) if existing else 1
        path = recovery_root / f"recovery-{next_num:04d}"
        path.mkdir()
        return path

    def _quarantine(self, recovery_dir: Path, src: Path, label: str) -> None:
        if not src.exists() and not src.is_symlink():
            return
        dest = recovery_dir / label
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))

    def recover(self) -> tuple[BookState, int]:
        """Run Appendix H recovery sequence; rebuild state from commits."""
        self.ensure_layout()
        head = self.read_head()
        committed = head.committed_page_count

        recovery_dir = self._next_recovery_dir()

        commits_root = self._path("commits")
        for name in os.listdir(commits_root):
            if PAGE_TMP_RE.match(name):
                self._quarantine(recovery_dir, commits_root / name, name)
            elif PAGE_COMMIT_RE.match(name):
                page_num = int(PAGE_COMMIT_RE.match(name).group(1))  # type: ignore[union-attr]
                if page_num > committed:
                    self._quarantine(recovery_dir, commits_root / name, name)

        for page_num in range(1, committed + 1):
            commit_dir = self.commit_dir_for(page_num)
            if not commit_dir.is_dir():
                raise ProcessingError(
                    code="invalid-run-layout",
                    message=f"missing accepted commit page-{page_num:04d}",
                )
            validate_commit(commit_dir)

        from bookextract.state import load_book_state_from_commits

        state = load_book_state_from_commits(self, committed)
        self.write_state_cache(state, committed)
        self._rebuild_failure_pointers()

        return state, committed

    def _rebuild_failure_pointers(self) -> None:
        failures_root = self._path("failures")
        if not failures_root.is_dir():
            return
        for page_dir in failures_root.iterdir():
            if not page_dir.is_dir() or not page_dir.name.startswith("page-"):
                continue
            failures = sorted(
                name
                for name in os.listdir(page_dir)
                if FAILURE_DIR_RE.match(name)
            )
            if not failures:
                continue
            latest_name = failures[-1]
            latest = {
                "latest_failure_pointer_format_version": 1,
                "failure_directory": latest_name,
            }
            write_json_atomic(page_dir / "latest.json", latest)

    def _next_failure_number(self, page_number: int) -> int:
        page_dir = self._path("failures", f"page-{page_number:04d}")
        page_dir.mkdir(parents=True, exist_ok=True)
        existing = [
            int(FAILURE_DIR_RE.match(name).group(1))  # type: ignore[union-attr]
            for name in os.listdir(page_dir)
            if FAILURE_DIR_RE.match(name)
        ]
        return (max(existing) + 1) if existing else 1

    def persist_failure(
        self,
        *,
        page_number: int,
        context: dict[str, Any],
        page_input: dict[str, Any],
        prompt: bytes,
        schema_ref: dict[str, Any],
        request_summary: dict[str, Any],
        error: dict[str, Any],
        attempts: dict[str, bytes] | None = None,
    ) -> FailureRecord:
        """Persist a complete failure directory atomically."""
        failure_num = self._next_failure_number(page_number)
        page_dir = self._path("failures", f"page-{page_number:04d}")
        tmp_dir = page_dir / f".failure-{failure_num:04d}.tmp"
        final_dir = page_dir / f"failure-{failure_num:04d}"

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        self.atomic_write_json(tmp_dir / "context.json", context)
        self.atomic_write_json(tmp_dir / "page-input.json", page_input)
        self.atomic_write_bytes(tmp_dir / "prompt.txt", prompt)
        self.atomic_write_json(tmp_dir / "schema-ref.json", schema_ref)
        self.atomic_write_json(tmp_dir / "request-summary.json", request_summary)
        self.atomic_write_json(tmp_dir / "error.json", error)

        if attempts:
            attempts_dir = tmp_dir / "attempts"
            attempts_dir.mkdir()
            for name, body in attempts.items():
                self.atomic_write_bytes(attempts_dir / name, body)

        os.replace(tmp_dir, final_dir)

        latest = {
            "latest_failure_pointer_format_version": 1,
            "failure_directory": final_dir.name,
        }
        write_json_atomic(page_dir / "latest.json", latest)
        return FailureRecord(failure_directory=final_dir, failure_number=failure_num)

    def _next_preflight_dir(self) -> tuple[Path, Path]:
        diagnostics_root = self._path("diagnostics")
        diagnostics_root.mkdir(parents=True, exist_ok=True)
        existing = [
            int(PREFLIGHT_DIR_RE.match(name).group(1))  # type: ignore[union-attr]
            for name in os.listdir(diagnostics_root)
            if PREFLIGHT_DIR_RE.match(name)
        ]
        next_num = (max(existing) + 1) if existing else 1
        final_dir = diagnostics_root / f"preflight-{next_num:04d}"
        tmp_dir = diagnostics_root / f".preflight-{next_num:04d}.tmp"
        return tmp_dir, final_dir

    def write_preflight_diagnostics(
        self,
        *,
        preflight: object,
        environment: InferenceEnvironment,
        smoke_files: dict[str, bytes],
    ) -> Path:
        """Persist thinking-smoke diagnostics atomically."""
        from bookextract.inference.llamacpp import PreflightResult

        if not isinstance(preflight, PreflightResult):
            raise TypeError("preflight must be PreflightResult")

        tmp_dir, final_dir = self._next_preflight_dir()
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        preflight_payload = {
            "identity": preflight.identity.model_dump(mode="json"),
            "capabilities": preflight.capabilities.model_dump(mode="json"),
        }
        self.atomic_write_json(tmp_dir / "preflight.json", preflight_payload)
        self.atomic_write_json(
            tmp_dir / "result.json",
            {
                "environment": environment.model_dump(mode="json"),
                "thinking_smoke": "passed",
            },
        )
        for name, content in smoke_files.items():
            self.atomic_write_bytes(tmp_dir / name, content)

        os.replace(tmp_dir, final_dir)
        return final_dir
