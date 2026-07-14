"""File fingerprinting for inference environment binding and relocation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from bookextract.models import FileFingerprint


def fingerprint_file(path: Path) -> FileFingerprint:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    stat = path.stat()
    return FileFingerprint(
        path=str(path.resolve()),
        size=size,
        mtime_ns=stat.st_mtime_ns,
        sha256=digest.hexdigest(),
    )
