"""Inference artifact invariant tests."""

from __future__ import annotations

import pytest

from bookextract.artifacts import (
    InferenceAttempt,
    PendingArtifact,
    RequestSnapshot,
    validate_artifact_filenames,
)
from bookextract.errors import ProcessingError


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


def test_validate_artifact_filenames_rejects_reserved_names() -> None:
    with pytest.raises(ProcessingError) as exc_info:
        validate_artifact_filenames(
            (
                PendingArtifact(
                    logical_name="prompt",
                    filename="page-assessment.json",
                    media_type="text/plain",
                    content=b"x",
                ),
            )
        )
    assert exc_info.value.code == "duplicate-artifact-name"


def test_validate_artifact_filenames_rejects_duplicates() -> None:
    artifact = PendingArtifact(
        logical_name="prompt",
        filename="prompt.txt",
        media_type="text/plain",
        content=b"x",
    )
    with pytest.raises(ProcessingError) as exc_info:
        validate_artifact_filenames((artifact, artifact))
    assert exc_info.value.code == "duplicate-artifact-name"
