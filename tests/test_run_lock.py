"""Tests for POSIX run lock."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

import pytest

from bookextract.errors import ProcessingError
from bookextract.run_lock import acquire_run_lock


def test_acquire_run_lock_creates_lock_file(tmp_path: Path) -> None:
    with acquire_run_lock(tmp_path):
        assert (tmp_path / ".bookextract.lock").is_file()


def test_second_acquire_fails_with_run_locked(tmp_path: Path) -> None:
    first = acquire_run_lock(tmp_path)
    fd = os.open(
        tmp_path / ".bookextract.lock",
        os.O_RDWR | os.O_NOFOLLOW,
    )
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)
    first.__exit__(None, None, None)

    with acquire_run_lock(tmp_path):
        pass


def test_run_locked_when_already_held(tmp_path: Path) -> None:
    with acquire_run_lock(tmp_path):
        with pytest.raises(ProcessingError, match="run-locked"):
            acquire_run_lock(tmp_path)
