"""Inference and interpretation artifact types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from bookextract.models import InterpretationProvenance, PageInterpretation

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PendingArtifact:
    logical_name: str
    media_type: str
    content: bytes
    filename: str | None = None


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
    attempts: tuple[InferenceAttempt, ...]


@dataclass(frozen=True, slots=True)
class InterpretationResult:
    interpretation: PageInterpretation
    provenance: InterpretationProvenance
    artifacts: tuple[PendingArtifact, ...] = ()
    failed_attempts: tuple[InferenceAttempt, ...] = ()
