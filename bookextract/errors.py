"""Error hierarchy and exit codes."""

from __future__ import annotations

from dataclasses import dataclass


class BookExtractError(Exception):
    code: str

    def __init__(self, code: str, message: str = "") -> None:
        self.code = code
        super().__init__(message or code)


class ProcessingError(BookExtractError):
    """Run-state or configuration failure (exit 13/14)."""


class StructuralError(BookExtractError):
    """Validation failure after inference (exit 11)."""


class ExternalToolError(BookExtractError):
    """Pandoc/EPUBCheck failure (exit 12)."""


@dataclass(frozen=True, slots=True)
class InferenceFailureContext:
    prompt: bytes
    request_summary: bytes
    schema_ref: bytes
    page_image_sha256: str
    wire_request_sha256: str | None


class InferenceError(BookExtractError):
    def __init__(
        self,
        code: str,
        *,
        retryable: bool,
        attempts_exhausted: bool,
        context: InferenceFailureContext,
        attempts: tuple[object, ...] = (),
        message: str = "",
    ) -> None:
        self.retryable = retryable
        self.attempts_exhausted = attempts_exhausted
        self.context = context
        self.attempts = attempts
        super().__init__(code, message)


RUN_STATE_CODES = frozenset(
    {
        "invalid-run-layout",
        "source-hash-mismatch",
        "config-drift",
        "schema-drift",
        "inference-environment-drift",
        "render-environment-drift",
        "static-context-impossible",
        "unsupported-multi-model-server",
        "unsupported-llama-server-contract",
        "unverifiable-server-model-path",
        "server-model-path-mismatch",
        "unsupported-thinking-control",
        "thinking-control-contract-drift",
        "token-counting-contract-drift",
        "invalid-commit-artifact-path",
        "unsupported-pandoc-defaults",
        "output-transaction-corruption",
    }
)
EXIT_SUCCESS = 0
EXIT_UNEXPECTED = 1
EXIT_ARGPARSE = 2
EXIT_INFERENCE = 10
EXIT_STRUCTURAL = 11
EXIT_EXTERNAL = 12
EXIT_RUN_STATE = 13
EXIT_PDF_STORAGE = 14


def exit_code_for_error(exc: BaseException) -> int:
    if isinstance(exc, InferenceError):
        return EXIT_INFERENCE
    if isinstance(exc, StructuralError):
        return EXIT_STRUCTURAL
    if isinstance(exc, ExternalToolError):
        return EXIT_EXTERNAL
    if isinstance(exc, ProcessingError):
        if exc.code in {"invalid-artifact-path", "invalid-source-pdf"}:
            return EXIT_PDF_STORAGE
        return EXIT_RUN_STATE
    return EXIT_UNEXPECTED
