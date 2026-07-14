"""Inference artifact invariant tests."""

from __future__ import annotations

import pytest

from bookextract.artifacts import InferenceAttempt, RequestSnapshot


def test_successful_attempt_cannot_carry_error() -> None:
    with pytest.raises(ValueError, match="error code"):
        InferenceAttempt(attempt_number=1, succeeded=True, error_code="x")


def test_failed_attempt_requires_error_code() -> None:
    with pytest.raises(ValueError, match="error code"):
        InferenceAttempt(attempt_number=1, succeeded=False)


def test_request_snapshot_stages() -> None:
    planned = RequestSnapshot(stage="planned")
    serialized = RequestSnapshot(stage="serialized", wire_request_sha256="abc")
    assert planned.wire_request_sha256 is None
    assert serialized.wire_request_sha256 == "abc"
