"""POSIX run-directory exclusive lock."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

from bookextract.errors import ProcessingError


def validate_original_path_components_without_following_symlinks(path: Path) -> None:
    """Reject symlink components in the user-supplied path before resolve()."""
    current = path
    parts: list[str] = []
    while True:
        parts.append(current.name if current.name else str(current))
        if current == current.parent:
            break
        current = current.parent
    parts.reverse()

    check = Path(path.anchor) if path.is_absolute() else Path(".")
    for part in parts[1:] if path.is_absolute() else parts:
        if part in ("", "."):
            continue
        check = check / part
        if check.is_symlink():
            raise ProcessingError(
                code="invalid-run-layout",
                message=f"symlink component in run path: {check}",
            )


class RunLock:
    def __init__(self, fd: int) -> None:
        self._fd = fd

    def __enter__(self) -> RunLock:
        return self

    def __exit__(self, *_exc: object) -> None:
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)


def acquire_run_lock(run_dir: Path) -> RunLock:
    validate_original_path_components_without_following_symlinks(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    lock_path = run_dir / ".bookextract.lock"
    fd = os.open(
        str(lock_path),
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
        0o600,
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise ProcessingError(code="run-locked") from None
    return RunLock(fd)
