"""First-process and later-process inference environment preparation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from bookextract.config import InferenceLocation
from bookextract.errors import ProcessingError
from bookextract.failure_persistence import persist_page_preparation_failure
from bookextract.fingerprints import fingerprint_file
from bookextract.inference.llamacpp import LlamaCppVisionClient
from bookextract.models import (
    FileFingerprint,
    InferenceEnvironment,
    PageContext,
    RenderedPage,
    ServerInferenceIdentity,
    ThinkingControlContract,
    TokenCountingContract,
)
from bookextract.schema import build_wire_response_format
from bookextract.storage import RunStore


def _assert_server_identity(
    current: ServerInferenceIdentity,
    stored: ServerInferenceIdentity,
) -> None:
    fields = (
        "llama_cpp_build",
        "model_alias",
        "context_size",
        "vision_supported",
        "chat_template_sha256",
    )
    for field in fields:
        if getattr(current, field) != getattr(stored, field):
            raise ProcessingError(
                code="inference-environment-drift",
                message=f"server identity drift: {field}",
            )


def _assert_contract_identity(
    contract: TokenCountingContract | ThinkingControlContract,
    identity: ServerInferenceIdentity,
) -> None:
    if contract.model_alias != identity.model_alias:
        raise ProcessingError(
            code="inference-environment-drift",
            message="contract model_alias drift",
        )
    if contract.llama_cpp_build != identity.llama_cpp_build:
        raise ProcessingError(
            code="inference-environment-drift",
            message="contract llama_cpp_build drift",
        )
    if contract.chat_template_sha256 != identity.chat_template_sha256:
        raise ProcessingError(
            code="inference-environment-drift",
            message="contract chat_template_sha256 drift",
        )


def _assert_file_fingerprint(
    stored: FileFingerprint | None,
    current: FileFingerprint,
    *,
    label: str,
) -> None:
    if stored is None:
        return
    if stored.sha256 and current.sha256 and stored.sha256 != current.sha256:
        raise ProcessingError(
            code="inference-environment-drift",
            message=f"{label} fingerprint drift",
        )


def _projector_binding(location: InferenceLocation) -> str:
    if location.projector_file_path is not None:
        return "operator-asserted"
    return "unavailable"


def prepare_inference_environment(
    *,
    store: RunStore,
    client: LlamaCppVisionClient,
    calibration_context: PageContext,
    calibration_prompt: str,
    render_calibration_page: Callable[[], RenderedPage] | None,
) -> InferenceEnvironment:
    """Run preflight, calibrate or verify contracts, smoke test, and bind the client."""
    preflight = client.preflight()
    location = store.load_inference_location()
    client.verify_model_path_binding(preflight, location.model_file_path)

    model_file = fingerprint_file(location.model_file_path)
    projector_file = (
        fingerprint_file(location.projector_file_path)
        if location.projector_file_path is not None
        else None
    )

    existing = store.load_inference_environment()
    response_format = build_wire_response_format(client._config.extraction.wire_schema_version)

    if existing is None:
        thinking_contract = client.calibrate_thinking_control(preflight)

        calibration_image: Path | None = None
        if render_calibration_page is not None:
            try:
                calibration_page = render_calibration_page()
            except ProcessingError as exc:
                if exc.code not in {"page-image-too-large", "page-render-failed"}:
                    raise
                persist_page_preparation_failure(
                    store=store,
                    page_index=0,
                    context=calibration_context.model_dump(mode="json"),
                    extraction_config=client._config.extraction,
                    error=exc,
                )
                raise
            calibration_image = calibration_page.image_path

        if calibration_image is None:
            raise ProcessingError(
                code="inference-environment-drift",
                message="calibration render callback required for first environment",
            )

        token_contract = client.discover_token_counting_contract(
            preflight,
            prompt=calibration_prompt,
            image_path=calibration_image,
            response_format=response_format,
            thinking_contract=thinking_contract,
        )
        environment = InferenceEnvironment(
            server=preflight.identity,
            model_file=model_file,
            projector_file=projector_file,
            model_binding_verified=True,
            projector_binding=_projector_binding(location),
            fingerprints_complete=True,
            token_counting_contract=token_contract,
            thinking_control_contract=thinking_contract,
        )
        smoke_files = client.run_thinking_smoke(thinking_contract)
        store.write_preflight_diagnostics(
            preflight=preflight,
            environment=environment,
            smoke_files=smoke_files,
        )
        store.write_inference_environment(environment)
        client.bind_environment(environment)
        return environment

    _assert_server_identity(preflight.identity, existing.server)
    _assert_contract_identity(existing.token_counting_contract, preflight.identity)
    _assert_contract_identity(existing.thinking_control_contract, preflight.identity)
    _assert_file_fingerprint(existing.model_file, model_file, label="model_file")
    if projector_file is not None:
        _assert_file_fingerprint(
            existing.projector_file,
            projector_file,
            label="projector_file",
        )

    smoke_files = client.run_thinking_smoke(existing.thinking_control_contract)
    store.write_preflight_diagnostics(
        preflight=preflight,
        environment=existing,
        smoke_files=smoke_files,
    )
    client.bind_environment(existing)
    return existing
