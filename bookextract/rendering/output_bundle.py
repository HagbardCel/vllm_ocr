"""Helpers for assembling publishable output bundles."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from bookextract.canonical import (
    normalize_publication_document_nfc,
    publication_fingerprint_hex,
    publication_identifier,
)
from bookextract.config import write_json_atomic
from bookextract.models import OutputFileEntry, OutputManifest, PublicationDocument, SourceMapEntry
from bookextract.output_paths import figure_asset_path, validate_output_tree
from bookextract.storage import RunStore


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect_commit_assets(store: RunStore) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    head = store.read_head().committed_page_count
    for page_number in range(1, head + 1):
        commit_dir = store.commit_dir_for(page_number)
        assets_dir = commit_dir / "assets"
        if not assets_dir.is_dir():
            continue
        for path in assets_dir.rglob("*.png"):
            if path.is_symlink() or not path.is_file():
                continue
            content = path.read_bytes()
            digest = _sha256_bytes(content)
            assets[figure_asset_path(digest)] = content
    return assets


def build_output_manifest(
    *,
    command: Literal["markdown", "epub"],
    committed_page_count: int,
    publication: PublicationDocument,
    source_map: list[SourceMapEntry],
    files: list[OutputFileEntry],
) -> OutputManifest:
    normalized = normalize_publication_document_nfc(publication)
    return OutputManifest(
        output_manifest_format_version=2,
        command=command,
        committed_page_count=committed_page_count,
        publication_identifier=publication_identifier(normalized),
        publication_fingerprint=publication_fingerprint_hex(normalized),
        source_map=sorted(source_map, key=lambda entry: entry.publication_block_index),
        files=files,
    )


def write_output_bundle(
    candidate_root: Path,
    *,
    command: Literal["markdown", "epub"],
    primary_name: str,
    primary_bytes: bytes,
    manifest: OutputManifest,
    assets: dict[str, bytes],
) -> None:
    primary_path = candidate_root / primary_name
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    primary_path.write_bytes(primary_bytes)

    entries = [
        OutputFileEntry(
            path=primary_name,
            sha256=_sha256_bytes(primary_bytes),
            size_bytes=len(primary_bytes),
        )
    ]
    for rel_path, content in sorted(assets.items()):
        dest = candidate_root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        entries.append(
            OutputFileEntry(
                path=rel_path,
                sha256=_sha256_bytes(content),
                size_bytes=len(content),
            )
        )

    manifest_payload = manifest.model_copy(update={"files": entries})
    write_json_atomic(
        candidate_root / "manifest.json",
        manifest_payload.model_dump(mode="json"),
    )
    validate_output_tree(
        candidate_root,
        expected_command=command,
        expected_committed_page_count=manifest.committed_page_count,
    )
