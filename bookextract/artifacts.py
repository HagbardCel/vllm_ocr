"""Inference and interpretation artifact types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from bookextract.errors import ProcessingError
from bookextract.models import InterpretationProvenance, PageInterpretation

T = TypeVar("T")

_RESERVED_COMMIT_FILENAMES = frozenset(
    {
        "manifest.json",
        "page-assessment.json",
        "page-context.json",
        "interpretation.json",
        "provenance.json",
        "assets",
        "attempts",
    }
)


@dataclass(frozen=True, slots=True)
class PendingArtifact:
    logical_name: str
    filename: str
    media_type: str
    content: bytes


def validate_artifact_filenames(artifacts: tuple[PendingArtifact, ...]) -> None:
    seen: set[str] = set()
    for artifact in artifacts:
        filename = artifact.filename
        if not filename or filename != filename.strip():
            raise ProcessingError(
                code="duplicate-artifact-name",
                message="artifact filename must be a non-empty basename",
            )
        if "/" in filename or "\\" in filename:
            raise ProcessingError(
                code="duplicate-artifact-name",
                message=f"artifact filename must not contain path separators: {filename!r}",
            )
        if filename in {".", ".."}:
            raise ProcessingError(
                code="duplicate-artifact-name",
                message=f"artifact filename is reserved: {filename!r}",
            )
        if filename in _RESERVED_COMMIT_FILENAMES:
            raise ProcessingError(
                code="duplicate-artifact-name",
                message=f"artifact filename collides with reserved commit name: {filename!r}",
            )
        if filename in seen:
            raise ProcessingError(
                code="duplicate-artifact-name",
                message=f"duplicate artifact filename: {filename!r}",
            )
        seen.add(filename)


@dataclass(frozen=True, slots=True)
class RequestSnapshot:
    stage: Literal["planned", "serialized"]
    wire_request_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class InferenceAttempt:
    attempt_number: int
    succeeded: bool
    status_code: int | None = None
    response_body: bytes | None = None
    content_type: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    elapsed_ms: float | None = None
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        if self.attempt_number < 1:
            raise ValueError("attempt_number must be positive")
        if self.succeeded:
            if self.error_code is not None:
                raise ValueError("successful attempt cannot have an error code")
            if self.error_message is not None:
                raise ValueError("successful attempt cannot have an error message")
            if self.response_body is not None:
                raise ValueError("successful body belongs in final_raw_body")
        elif self.error_code is None:
            raise ValueError("failed attempt requires an error code")


@dataclass(frozen=True, slots=True)
class InferenceResult(Generic[T]):
    value: T
    final_raw_body: bytes
    request_snapshot: RequestSnapshot
    request_summary: dict[str, object]
    attempts: tuple[InferenceAttempt, ...]


@dataclass(frozen=True, slots=True)
class InterpretationResult:
    interpretation: PageInterpretation
    provenance: InterpretationProvenance
    artifacts: tuple[PendingArtifact, ...] = ()
    failed_attempts: tuple[InferenceAttempt, ...] = ()
