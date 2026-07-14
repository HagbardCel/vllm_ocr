"""Output path helpers and publishable tree validation."""

from __future__ import annotations

import hashlib
import re
import stat
from pathlib import Path
from typing import Literal

from bookextract.errors import ProcessingError
from bookextract.models import OutputManifest
from bookextract.storage import validate_commit_relative_path

__all__ = [
    "figure_asset_path",
    "parse_asset_sha256_from_path",
    "validate_output_relative_path",
    "validate_output_tree",
]

_ASSET_PATH_RE = re.compile(r"^assets/([0-9a-f]{64})\.png$")


def figure_asset_path(sha256: str) -> str:
    return f"assets/{sha256}.png"


def parse_asset_sha256_from_path(path: str) -> str | None:
    match = _ASSET_PATH_RE.match(path)
    if match is None:
        return None
    return match.group(1)


def validate_output_relative_path(rel_path: str) -> None:
    validate_commit_relative_path(rel_path)
    if rel_path == "manifest.json":
        raise ProcessingError(
            code="invalid-artifact-path",
            message="manifest.json cannot appear in file inventory",
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _primary_filename(command: Literal["markdown", "epub"]) -> str:
    return "book.md" if command == "markdown" else "book.epub"


def _walk_tree(root: Path) -> tuple[set[str], list[str]]:
    """Return (regular_file_relpaths, problems). Does not follow symlinks."""
    files: set[str] = set()
    problems: list[str] = []

    def visit(dir_path: Path) -> None:
        if dir_path.is_symlink():
            problems.append(f"symlink directory: {dir_path.relative_to(root).as_posix()}")
            return
        if not dir_path.is_dir():
            problems.append(f"non-directory: {dir_path}")
            return
        try:
            entries = list(dir_path.iterdir())
        except OSError as exc:
            problems.append(str(exc))
            return
        if not entries and dir_path != root:
            problems.append(f"unexpected empty directory: {dir_path.relative_to(root).as_posix()}")
            return
        for entry in entries:
            rel = entry.relative_to(root).as_posix()
            if entry.is_symlink():
                problems.append(f"symlink: {rel}")
                continue
            mode = entry.stat().st_mode
            if stat.S_ISREG(mode):
                files.add(rel)
            elif stat.S_ISDIR(mode):
                visit(entry)
            else:
                problems.append(f"non-regular file: {rel}")

    if root.is_symlink():
        problems.append("candidate root is a symlink")
        return files, problems
    if not root.is_dir():
        problems.append("candidate root is not a directory")
        return files, problems
    visit(root)
    return files, problems


def validate_output_tree(
    path: Path,
    *,
    expected_command: Literal["markdown", "epub"],
    expected_committed_page_count: int | None,
) -> OutputManifest:
    """Validate a publishable output tree and return its manifest."""
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ProcessingError(
            code="invalid-run-layout",
            message="missing or invalid manifest.json in output tree",
        )

    try:
        manifest = OutputManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise ProcessingError(
            code="invalid-run-layout",
            message=f"invalid output manifest: {exc}",
        ) from exc

    if manifest.command != expected_command:
        raise ProcessingError(
            code="invalid-run-layout",
            message=f"manifest command {manifest.command!r} != {expected_command!r}",
        )

    if (
        expected_committed_page_count is not None
        and manifest.committed_page_count != expected_committed_page_count
    ):
        raise ProcessingError(
            code="invalid-run-layout",
            message=(
                f"manifest committed_page_count {manifest.committed_page_count} "
                f"!= expected {expected_committed_page_count}"
            ),
        )

    primary = _primary_filename(expected_command)
    declared_paths: list[str] = []
    seen: set[str] = set()
    primary_count = 0
    for entry in manifest.files:
        validate_output_relative_path(entry.path)
        if entry.path in seen:
            raise ProcessingError(
                code="invalid-artifact-path",
                message=f"duplicate manifest path: {entry.path}",
            )
        seen.add(entry.path)
        declared_paths.append(entry.path)
        if entry.path == primary:
            primary_count += 1
        asset_sha = parse_asset_sha256_from_path(entry.path)
        if asset_sha is not None and asset_sha != entry.sha256:
            raise ProcessingError(
                code="invalid-run-layout",
                message=f"asset filename hash mismatch for {entry.path}",
            )

    if primary_count != 1:
        raise ProcessingError(
            code="invalid-run-layout",
            message=f"expected exactly one {primary!r} in manifest.files",
        )

    for entry in manifest.files:
        if entry.path == primary:
            continue
        if parse_asset_sha256_from_path(entry.path) is None:
            raise ProcessingError(
                code="invalid-artifact-path",
                message=f"non-asset path in manifest.files: {entry.path!r}",
            )

    expected_files = {"manifest.json", *declared_paths}
    actual_files, problems = _walk_tree(path)
    if problems:
        raise ProcessingError(
            code="invalid-run-layout",
            message="; ".join(problems),
        )
    if actual_files != expected_files:
        raise ProcessingError(
            code="invalid-run-layout",
            message=(
                f"output tree closure mismatch: "
                f"extra={sorted(actual_files - expected_files)} "
                f"missing={sorted(expected_files - actual_files)}"
            ),
        )

    for entry in manifest.files:
        file_path = path / entry.path
        if file_path.is_symlink() or not file_path.is_file():
            raise ProcessingError(
                code="invalid-run-layout",
                message=f"missing manifest file: {entry.path}",
            )
        size = file_path.stat().st_size
        if size != entry.size_bytes:
            raise ProcessingError(
                code="invalid-run-layout",
                message=f"size mismatch for {entry.path}",
            )
        if _sha256_file(file_path) != entry.sha256:
            raise ProcessingError(
                code="invalid-run-layout",
                message=f"hash mismatch for {entry.path}",
            )
        parent = file_path.parent
        rel_parent = parent.relative_to(path)
        parts = rel_parent.parts if rel_parent.parts != (".",) else ()
        current = path
        for part in parts:
            current = current / part
            if current.is_symlink() or not current.is_dir():
                raise ProcessingError(
                    code="invalid-artifact-path",
                    message=f"invalid path component for {entry.path}",
                )

    return manifest
