"""llama.cpp vision client with preflight, contracts, and context budgeting."""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bookextract.artifacts import InferenceAttempt, InferenceResult, RequestSnapshot
from bookextract.canonical import serialize_wire_request, sha256_hex
from bookextract.config import ProcessingConfig
from bookextract.errors import InferenceError, InferenceFailureContext, ProcessingError
from bookextract.models import (
    ApplyTemplateTokenizeContract,
    ContextBudgetResult,
    EstimateOnlyContract,
    InferenceEnvironment,
    MultimodalInputTokensContract,
    ServerInferenceIdentity,
    ServerInvocationCapabilities,
    TextOnlyInputTokensContract,
    ThinkingControlContract,
    TokenCountingContract,
    TokenCountingMode,
)
from bookextract.schema import build_wire_response_format, load_wire_schema, normalize_llama_schema
from bookextract.wire import VlmPageResponse

ResponseT = TypeVar("ResponseT", bound=BaseModel)

_TRANSIENT_STATUS_CODES = frozenset({429, 502, 503, 504})
_THINKING_FORMAT_CANDIDATES: tuple[Literal["none", "deepseek", "deepseek-legacy"], ...] = (
    "none",
    "deepseek",
    "deepseek-legacy",
)
_SMOKE_PROMPT = "Reply with the JSON schema only. No reasoning."
_ESTIMATE_CHARS_PER_TOKEN = 4


class LlamaWireModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)


class LlamaAssistantMessage(LlamaWireModel):
    content: str | None = None
    reasoning_content: str | None = None


class LlamaChatCompletionChoice(LlamaWireModel):
    index: int
    message: LlamaAssistantMessage
    finish_reason: str | None = None


class LlamaChatCompletionResponse(LlamaWireModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[LlamaChatCompletionChoice]
    usage: dict[str, int] | None = None


class LlamaModelEntry(LlamaWireModel):
    id: str


class LlamaModelsResponse(LlamaWireModel):
    data: list[LlamaModelEntry]


class LlamaPropsResponse(LlamaWireModel):
    build_commit: str | None = Field(default=None, alias="build")
    build_number: int | None = None
    build_info: str | None = None
    model_path: str | None = None
    default_generation_settings: dict[str, Any] | None = None
    chat_template: str | None = None
    chat_template_caps: dict[str, Any] | None = None
    media_marker: str | None = None
    modalities: dict[str, Any] | None = None
    total_slots: int | None = None

    model_config = ConfigDict(extra="ignore", strict=True, populate_by_name=True)


class LlamaInputTokensResponse(LlamaWireModel):
    input_tokens: int


@dataclass(frozen=True, slots=True)
class PreflightResult:
    identity: ServerInferenceIdentity
    capabilities: ServerInvocationCapabilities
    raw_health: bytes
    raw_models: bytes
    raw_props: bytes


class LlamaCppVisionClient:
    def __init__(
        self,
        config: ProcessingConfig,
        *,
        environment: InferenceEnvironment | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self._environment = environment
        self._owns_client = client is None
        base_url = config.process.llama_base_url.rstrip("/")
        self._client = client or httpx.Client(
            base_url=base_url,
            timeout=config.process.request_timeout_seconds,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> LlamaCppVisionClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def environment(self) -> InferenceEnvironment | None:
        return self._environment

    def bind_environment(self, environment: InferenceEnvironment) -> None:
        self._environment = environment

    def preflight(self) -> PreflightResult:
        health = self._get_bytes("/health")
        models_raw = self._get_bytes("/v1/models")
        props_raw = self._get_bytes("/props")

        try:
            models = LlamaModelsResponse.model_validate_json(models_raw)
        except ValidationError as exc:
            raise ProcessingError(
                code="unsupported-llama-server-contract",
                message=f"invalid /v1/models response: {exc}",
            ) from exc

        if len(models.data) != 1:
            raise ProcessingError(code="unsupported-multi-model-server")

        model_alias = self._config.extraction.model_alias
        if models.data[0].id != model_alias:
            raise ProcessingError(
                code="unsupported-llama-server-contract",
                message=f"expected model alias {model_alias!r}, got {models.data[0].id!r}",
            )

        try:
            props = LlamaPropsResponse.model_validate_json(props_raw)
        except ValidationError as exc:
            raise ProcessingError(
                code="unsupported-llama-server-contract",
                message=f"invalid /props response: {exc}",
            ) from exc

        llama_cpp_build = self._llama_cpp_build_from_props(props)
        if not llama_cpp_build:
            raise ProcessingError(
                code="unsupported-llama-server-contract",
                message="missing llama.cpp build identity in /props",
            )

        if not props.model_path:
            raise ProcessingError(
                code="unsupported-llama-server-contract",
                message="missing model_path in /props",
            )

        self._require_vision_modalities(props)

        context_size = self._context_size_from_props(props)
        self.check_static_context_feasibility(context_size)

        chat_template_sha256 = sha256_hex((props.chat_template or "").encode("utf-8"))
        identity = ServerInferenceIdentity(
            llama_cpp_build=llama_cpp_build,
            model_alias=model_alias,
            context_size=context_size,
            vision_supported=True,
            chat_template_sha256=chat_template_sha256,
            server_reported_model_path=props.model_path,
        )
        capabilities = ServerInvocationCapabilities(
            media_marker=self._media_marker(props),
            chat_template_caps=self._chat_template_caps(props),
        )
        return PreflightResult(
            identity=identity,
            capabilities=capabilities,
            raw_health=health,
            raw_models=models_raw,
            raw_props=props_raw,
        )

    def verify_model_path_binding(
        self,
        preflight: PreflightResult,
        configured_model_path: Path,
    ) -> None:
        server_model_path = Path(preflight.identity.server_reported_model_path)
        if not server_model_path.is_absolute():
            raise ProcessingError(code="unverifiable-server-model-path")
        if server_model_path.resolve() != configured_model_path.resolve():
            raise ProcessingError(code="server-model-path-mismatch")

    def check_static_context_feasibility(self, context_size: int) -> None:
        extraction = self._config.extraction
        required = extraction.max_tokens + extraction.context_safety_margin_tokens
        if context_size <= required:
            raise ProcessingError(code="static-context-impossible")

    def discover_token_counting_contract(
        self,
        preflight: PreflightResult,
        *,
        prompt: str,
        image_path: Path,
        response_format: dict[str, object],
        thinking_contract: ThinkingControlContract,
    ) -> TokenCountingContract:
        identity = preflight.identity
        text_payload = self._build_chat_payload(
            prompt=prompt,
            image_path=None,
            response_format=response_format,
            thinking_contract=thinking_contract,
            include_image=False,
        )
        image_payload = self._build_chat_payload(
            prompt=prompt,
            image_path=image_path,
            response_format=response_format,
            thinking_contract=thinking_contract,
            include_image=True,
        )

        text_count = self._probe_input_tokens(text_payload, discovery=True)
        if text_count is not None:
            image_count = self._probe_input_tokens(image_payload, discovery=True)
            if image_count is not None:
                mode: TokenCountingMode
                if image_count > text_count:
                    mode = "chat-input-tokens-multimodal"
                    contract: TokenCountingContract = MultimodalInputTokensContract(
                        mode=mode,
                        model_alias=identity.model_alias,
                        llama_cpp_build=identity.llama_cpp_build,
                        chat_template_sha256=identity.chat_template_sha256,
                    )
                else:
                    mode = "chat-input-tokens-text-only"
                    contract = TextOnlyInputTokensContract(
                        mode=mode,
                        image_token_policy="configured-reserve",
                        model_alias=identity.model_alias,
                        llama_cpp_build=identity.llama_cpp_build,
                        chat_template_sha256=identity.chat_template_sha256,
                    )
                return contract

        if self._apply_template_tokenize_supported(
            preflight,
            text_payload=text_payload,
        ):
            return self._build_apply_template_contract(preflight)

        return EstimateOnlyContract(
            mode="estimate-only",
            model_alias=identity.model_alias,
            llama_cpp_build=identity.llama_cpp_build,
            chat_template_sha256=identity.chat_template_sha256,
        )

    def calibrate_thinking_control(self, preflight: PreflightResult) -> ThinkingControlContract:
        identity = preflight.identity
        response_format = build_wire_response_format(self._config.extraction.wire_schema_version)
        applied_template_supported = self._applied_template_probe_supported(preflight.capabilities)

        for reasoning_format in _THINKING_FORMAT_CANDIDATES:
            contract = ThinkingControlContract(
                reasoning_format=reasoning_format,
                applied_template_probe_supported=applied_template_supported,
                model_alias=identity.model_alias,
                llama_cpp_build=identity.llama_cpp_build,
                chat_template_sha256=identity.chat_template_sha256,
            )
            try:
                self._run_thinking_calibration_probes(contract, response_format)
            except _ThinkingCandidateRejected:
                continue
            except InferenceError:
                raise
            return contract

        raise ProcessingError(code="unsupported-thinking-control")

    def run_thinking_smoke(self, contract: ThinkingControlContract) -> dict[str, bytes]:
        response_format = build_wire_response_format(self._config.extraction.wire_schema_version)
        payload = self._build_chat_payload(
            prompt=_SMOKE_PROMPT,
            image_path=None,
            response_format=response_format,
            thinking_contract=contract,
            include_image=False,
        )
        wire_body = serialize_wire_request(payload)
        try:
            response = self._post_completion(wire_body, allow_retry=True)
        except InferenceError:
            raise
        except ProcessingError as exc:
            raise ProcessingError(
                code="thinking-control-contract-drift",
                message=exc.code,
            ) from exc

        try:
            self._enforce_non_thinking_response(response, contract, smoke=True)
        except ProcessingError as exc:
            raise ProcessingError(
                code="thinking-control-contract-drift",
                message=exc.code,
            ) from exc
        try:
            content = response.choices[0].message.content or ""
            VlmPageResponse.model_validate_json(content)
        except (ValidationError, IndexError) as exc:
            raise ProcessingError(code="thinking-control-contract-drift") from exc

        return {
            "thinking-smoke-request.json": wire_body,
            "thinking-smoke-response.json": response.model_dump_json().encode("utf-8"),
        }

    def generate_structured(
        self,
        *,
        image_path: Path,
        page_image_sha256: str,
        prompt: str,
        response_model: type[ResponseT],
        schema_ref: bytes,
    ) -> InferenceResult[ResponseT]:
        if self._environment is None:
            raise ProcessingError(
                code="inference-environment-drift",
                message="environment not bound",
            )

        response_format = build_wire_response_format(self._config.extraction.wire_schema_version)
        thinking_contract = self._environment.thinking_control_contract
        payload = self._build_chat_payload(
            prompt=prompt,
            image_path=image_path,
            response_format=response_format,
            thinking_contract=thinking_contract,
            include_image=True,
        )
        request_summary = self._build_request_summary_base(payload)

        try:
            budget = self._enforce_context_budget(
                payload=payload,
                contract=self._environment.token_counting_contract,
                context_size=self._environment.server.context_size,
            )
        except InferenceError:
            raise
        except ProcessingError as exc:
            if exc.code == "token-counting-contract-drift":
                raise
            raise InferenceError(
                code=exc.code,
                retryable=False,
                attempts_exhausted=False,
                context=self._failure_context(
                    prompt=prompt,
                    request_summary=request_summary,
                    schema_ref=schema_ref,
                    page_image_sha256=page_image_sha256,
                    wire_request_sha256=None,
                ),
                message=str(exc),
            ) from exc

        request_summary["context_budget"] = budget.model_dump(mode="json")
        wire_body = serialize_wire_request(payload)
        wire_request_sha256 = sha256_hex(wire_body)
        sanitized_summary = {
            **request_summary,
            "stage": "serialized",
            "wire_request_sha256": wire_request_sha256,
            "image_sha256": page_image_sha256,
        }
        failure_context = self._failure_context(
            prompt=prompt,
            request_summary=request_summary,
            schema_ref=schema_ref,
            page_image_sha256=page_image_sha256,
            wire_request_sha256=wire_request_sha256,
        )

        if budget.required_tokens > budget.context_size:
            raise InferenceError(
                code="context-budget-exceeded",
                retryable=False,
                attempts_exhausted=False,
                context=failure_context,
                message="projected input exceeds model context",
            )

        attempts: list[InferenceAttempt] = []
        max_attempts = self._config.process.max_attempts
        server_error_retries_used = 0
        for attempt_number in range(1, max_attempts + 1):
            started = time.monotonic()
            try:
                parsed, status_code, raw_body = self._complete_with_raw(wire_body)
            except _TransportFailure as exc:
                elapsed_ms = (time.monotonic() - started) * 1000
                attempts.append(
                    InferenceAttempt(
                        attempt_number=attempt_number,
                        succeeded=False,
                        status_code=exc.status_code,
                        response_body=exc.body,
                        content_type=exc.content_type,
                        error_code=exc.error_code,
                        error_message=exc.message,
                        elapsed_ms=elapsed_ms,
                    )
                )
                if self._may_retry(
                    exc.error_code,
                    attempt_number=attempt_number,
                    max_attempts=max_attempts,
                    server_error_retries_used=server_error_retries_used,
                ):
                    if exc.error_code == "http-server-error":
                        server_error_retries_used += 1
                    self._sleep_backoff(attempt_number, exc.retry_after_seconds)
                    continue
                break
            except _CompletionFailure as exc:
                if exc.status_code is None and exc.response_body is None:
                    raise
                elapsed_ms = (time.monotonic() - started) * 1000
                attempts.append(
                    InferenceAttempt(
                        attempt_number=attempt_number,
                        succeeded=False,
                        status_code=exc.status_code,
                        response_body=exc.response_body,
                        content_type=exc.content_type,
                        error_code=exc.code,
                        error_message=str(exc),
                        elapsed_ms=elapsed_ms,
                    )
                )
                break

            elapsed_ms = (time.monotonic() - started) * 1000
            finish_reason = (
                parsed.choices[0].finish_reason if parsed.choices else None
            )
            try:
                self._enforce_non_thinking_response(parsed, thinking_contract)
                value = self._parse_assistant_content(
                    parsed,
                    response_model=response_model,
                )
            except InferenceError:
                raise
            except _CompletionFailure as exc:
                attempts.append(
                    InferenceAttempt(
                        attempt_number=attempt_number,
                        succeeded=False,
                        status_code=status_code,
                        response_body=raw_body,
                        content_type="application/json",
                        error_code=exc.code,
                        error_message=str(exc),
                        elapsed_ms=elapsed_ms,
                        finish_reason=finish_reason,
                    )
                )
                break

            attempts.append(
                InferenceAttempt(
                    attempt_number=attempt_number,
                    succeeded=True,
                    status_code=status_code,
                    elapsed_ms=elapsed_ms,
                    finish_reason=finish_reason,
                )
            )
            return InferenceResult(
                value=value,
                final_raw_body=raw_body,
                request_snapshot=RequestSnapshot(
                    stage="serialized",
                    wire_request_sha256=wire_request_sha256,
                ),
                request_summary=sanitized_summary,
                attempts=tuple(attempts),
            )

        raise InferenceError(
            code=attempts[-1].error_code or "transport-timeout",
            retryable=self._may_retry(
                attempts[-1].error_code or "transport-timeout",
                attempt_number=len(attempts),
                max_attempts=max_attempts,
                server_error_retries_used=server_error_retries_used,
            ),
            attempts_exhausted=len(attempts) >= max_attempts,
            context=failure_context,
            attempts=tuple(attempts),
            message=attempts[-1].error_message or "completion failed",
        )

    def _enforce_context_budget(
        self,
        *,
        payload: dict[str, Any],
        contract: TokenCountingContract,
        context_size: int,
    ) -> ContextBudgetResult:
        extraction = self._config.extraction
        image_reserve = 0
        multimodal_included = False
        exact = False
        counted: int | None = None

        if contract.mode == "chat-input-tokens-multimodal":
            counted = self._count_with_frozen_contract(payload, contract)
            multimodal_included = True
            exact = True
        elif contract.mode == "chat-input-tokens-text-only":
            counted = self._count_with_frozen_contract(payload, contract)
            image_reserve = self._effective_image_tokens()
            exact = True
        elif contract.mode == "apply-template-tokenize":
            counted = self._count_with_frozen_contract(payload, contract)
            image_reserve = self._effective_image_tokens()
            exact = True
        elif contract.mode == "estimate-only":
            counted = self._estimate_tokens(payload)
            image_reserve = self._effective_image_tokens()
            exact = False

        if counted is None:
            raise ProcessingError(code="token-counting-contract-drift")

        budget = ContextBudgetResult(
            counted_input_tokens=counted,
            image_tokens_reserved=image_reserve,
            output_tokens_reserved=extraction.max_tokens,
            safety_margin_tokens=extraction.context_safety_margin_tokens,
            context_size=context_size,
            counting_mode=contract.mode,
            exact_for_projected_input=exact,
            multimodal_count_included=multimodal_included,
        )
        if (
            not exact
            and budget.required_tokens > context_size * 0.85
        ):
            raise InferenceError(
                code="context-budget-indeterminate",
                retryable=False,
                attempts_exhausted=False,
                context=self._failure_context(
                    prompt=str(payload.get("messages", "")),
                    request_summary=self._build_request_summary_base(payload),
                    schema_ref=b"{}",
                    page_image_sha256="",
                    wire_request_sha256=None,
                ),
            )
        return budget

    def _count_with_frozen_contract(
        self,
        payload: dict[str, Any],
        contract: TokenCountingContract,
    ) -> int | None:
        discovery = self._environment is None
        if contract.mode in {"chat-input-tokens-multimodal", "chat-input-tokens-text-only"}:
            return self._probe_input_tokens(payload, discovery=discovery)
        if contract.mode == "apply-template-tokenize":
            return self._count_apply_template_tokenize(payload, contract, discovery=discovery)
        if contract.mode == "estimate-only":
            return self._estimate_tokens(payload)
        return None

    def _probe_input_tokens(self, payload: dict[str, Any], *, discovery: bool) -> int | None:
        wire_body = serialize_wire_request(payload)

        def attempt() -> int | None:
            response = self._client.post(
                "/v1/chat/completions/input_tokens",
                content=wire_body,
                headers={"Content-Type": "application/json"},
            )
            if response.status_code == 404:
                return None
            if response.status_code == 400:
                if discovery:
                    raise ProcessingError(
                        code="token-counting-calibration-failed",
                        message="input_tokens rejected production request shape",
                    )
                raise ProcessingError(code="token-counting-contract-drift")
            if response.status_code in _TRANSIENT_STATUS_CODES:
                raise _ProbeTransient(response.status_code)
            if response.status_code >= 500:
                raise _ProbeTransient(response.status_code)
            if response.status_code != 200:
                if discovery:
                    raise ProcessingError(
                        code="token-counting-calibration-failed",
                        message=f"input_tokens returned HTTP {response.status_code}",
                    )
                raise ProcessingError(code="token-counting-contract-drift")
            try:
                parsed = LlamaInputTokensResponse.model_validate_json(response.content)
            except ValidationError:
                if discovery:
                    raise ProcessingError(
                        code="token-counting-calibration-failed",
                        message="input_tokens returned malformed successful response",
                    ) from None
                raise ProcessingError(code="token-counting-contract-drift") from None
            return parsed.input_tokens

        return self._run_budget_probe(attempt, discovery=discovery)

    def _count_apply_template_tokenize(
        self,
        payload: dict[str, Any],
        contract: ApplyTemplateTokenizeContract,
        *,
        discovery: bool,
    ) -> int | None:
        messages = self._project_text_only_messages(payload)
        apply_body: dict[str, Any] = {"messages": messages}
        if contract.apply_template_request_mode == "messages-plus-chat-template-kwargs":
            apply_body["chat_template_kwargs"] = {"enable_thinking": False}

        prompt = self._request_apply_template(
            apply_body,
            discovery=discovery,
            extended=contract.apply_template_request_mode
            == "messages-plus-chat-template-kwargs",
        )
        if prompt is None:
            return None
        return self._request_tokenize(
            prompt,
            add_special=contract.tokenize_add_special,
            parse_special=contract.tokenize_parse_special,
            discovery=discovery,
        )

    def _project_text_only_messages(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for message in cast(list[dict[str, Any]], payload.get("messages", [])):
            content = message.get("content")
            if isinstance(content, list):
                text_only_parts = [
                    part
                    for part in cast(list[dict[str, Any]], content)
                    if part.get("type") != "image_url"
                ]
                messages.append({**message, "content": text_only_parts})
            else:
                messages.append({**message, "content": content})
        return messages

    def _text_only_messages(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return self._project_text_only_messages(payload)

    def _run_budget_probe(self, fn: Any, *, discovery: bool) -> int | None:
        for probe_attempt in range(2):
            try:
                result = fn()
                return cast(int, result)
            except _ProbeTransient:
                if probe_attempt == 0:
                    time.sleep(self._config.process.retry_backoff_seconds)
                    continue
                self._raise_counting_probe_transient(discovery)
            except httpx.TimeoutException as exc:
                if probe_attempt == 0:
                    time.sleep(self._config.process.retry_backoff_seconds)
                    continue
                self._raise_counting_probe_transient(discovery, from_exc=exc)
            except httpx.TransportError as exc:
                if probe_attempt == 0:
                    time.sleep(self._config.process.retry_backoff_seconds)
                    continue
                self._raise_counting_probe_transient(discovery, from_exc=exc)
        return None

    def _raise_counting_probe_transient(
        self,
        discovery: bool,
        *,
        from_exc: BaseException | None = None,
    ) -> None:
        if discovery:
            raise ProcessingError(
                code="token-counting-calibration-failed",
                message="token-counting endpoint remained transiently unavailable",
            ) from from_exc
        raise InferenceError(
            code="context-budget-probe-unavailable",
            retryable=True,
            attempts_exhausted=False,
            context=self._empty_failure_context(),
        ) from from_exc

    def _run_thinking_calibration_probes(
        self,
        contract: ThinkingControlContract,
        response_format: dict[str, object],
    ) -> None:
        if contract.applied_template_probe_supported:
            try:
                self._run_applied_template_probe(contract)
            except _ThinkingCandidateRejected as exc:
                raise exc
            except InferenceError:
                raise

        raw_payload = self._build_chat_payload(
            prompt="Say hello in one short sentence.",
            image_path=None,
            response_format=None,
            thinking_contract=contract,
            include_image=False,
        )
        raw_response = self._post_completion(
            serialize_wire_request(raw_payload),
            allow_retry=False,
        )
        if raw_response.choices[0].message.reasoning_content:
            raise _ThinkingCandidateRejected()

        production_payload = self._build_chat_payload(
            prompt=_SMOKE_PROMPT,
            image_path=None,
            response_format=response_format,
            thinking_contract=contract,
            include_image=False,
        )
        production_response = self._post_completion(
            serialize_wire_request(production_payload),
            allow_retry=False,
        )
        try:
            self._enforce_non_thinking_response(production_response, contract)
            content = production_response.choices[0].message.content or ""
            VlmPageResponse.model_validate_json(content)
        except (ProcessingError, ValidationError, IndexError) as exc:
            raise _ThinkingCandidateRejected() from exc

    def _run_applied_template_probe(self, contract: ThinkingControlContract) -> None:
        payload = {
            "model": self._config.extraction.model_alias,
            "messages": [{"role": "user", "content": "probe"}],
            "chat_template_kwargs": {"enable_thinking": contract.enable_thinking},
            "reasoning_format": contract.reasoning_format,
        }
        try:
            response = self._client.post(
                "/apply-template",
                content=serialize_wire_request(payload),
                headers={"Content-Type": "application/json"},
            )
        except httpx.TimeoutException as exc:
            raise InferenceError(
                code="thinking-smoke-unavailable",
                retryable=True,
                attempts_exhausted=False,
                context=self._empty_failure_context(),
            ) from exc
        except httpx.TransportError as exc:
            raise InferenceError(
                code="thinking-smoke-unavailable",
                retryable=True,
                attempts_exhausted=False,
                context=self._empty_failure_context(),
            ) from exc
        if response.status_code == 404:
            return
        if response.status_code in _TRANSIENT_STATUS_CODES or response.status_code >= 500:
            raise InferenceError(
                code="thinking-smoke-unavailable",
                retryable=True,
                attempts_exhausted=False,
                context=self._empty_failure_context(),
            )
        if response.status_code != 200:
            raise _ThinkingCandidateRejected()

    def _post_completion(
        self,
        wire_body: bytes,
        *,
        allow_retry: bool,
    ) -> LlamaChatCompletionResponse:
        max_attempts = 2 if allow_retry else 1
        server_error_retries_used = 0

        for attempt_number in range(1, max_attempts + 1):
            try:
                response, _, _ = self._complete_with_raw(wire_body)
                return response
            except _TransportFailure as exc:
                if (
                    attempt_number < max_attempts
                    and self._may_retry(
                        exc.error_code,
                        attempt_number=attempt_number,
                        max_attempts=max_attempts,
                        server_error_retries_used=server_error_retries_used,
                    )
                ):
                    if exc.error_code == "http-server-error":
                        server_error_retries_used += 1
                    self._sleep_backoff(attempt_number, exc.retry_after_seconds)
                    continue

                if exc.error_code in {
                    "transport-timeout",
                    "transport-error",
                    "response-body-truncated",
                    "http-retryable",
                    "http-server-error",
                }:
                    raise InferenceError(
                        code="thinking-smoke-unavailable",
                        retryable=True,
                        attempts_exhausted=attempt_number >= max_attempts,
                        context=self._empty_failure_context(),
                    ) from exc

                raise ProcessingError(
                    code="thinking-control-contract-drift",
                    message=exc.error_code,
                ) from exc
            except _CompletionFailure as exc:
                raise ProcessingError(
                    code="thinking-control-contract-drift",
                    message=exc.code,
                ) from exc

        raise ProcessingError(code="thinking-control-contract-drift")

    def _perform_http_request(self, wire_body: bytes) -> tuple[httpx.Response, bytes]:
        try:
            response = self._client.post(
                "/v1/chat/completions",
                content=wire_body,
                headers={"Content-Type": "application/json"},
            )
        except httpx.RemoteProtocolError as exc:
            raise _TransportFailure(
                error_code="response-body-truncated",
                message=str(exc),
            ) from exc
        except httpx.TimeoutException as exc:
            raise _TransportFailure(
                error_code="transport-timeout",
                message="request timed out",
            ) from exc
        except httpx.TransportError as exc:
            raise _TransportFailure(
                error_code="transport-error",
                message=str(exc),
            ) from exc
        return response, response.content

    def _classify_http_status(self, status_code: int) -> str | None:
        if status_code == 200:
            return None
        if status_code in _TRANSIENT_STATUS_CODES:
            return "http-retryable"
        if status_code >= 500:
            return "http-server-error"
        return "http-client-error"

    def _complete_with_raw(
        self,
        wire_body: bytes,
    ) -> tuple[LlamaChatCompletionResponse, int, bytes]:
        response, raw_body = self._perform_http_request(wire_body)
        status_failure = self._classify_http_status(response.status_code)
        if status_failure is not None:
            retry_after = self._parse_retry_after(response.headers.get("Retry-After"))
            raise _TransportFailure(
                error_code=status_failure,
                message=f"HTTP {response.status_code}",
                status_code=response.status_code,
                body=raw_body,
                content_type=response.headers.get("Content-Type"),
                retry_after_seconds=retry_after,
            )

        content_type = response.headers.get("Content-Type")

        try:
            envelope_data = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise _CompletionFailure(
                "invalid-http-response",
                "response body is not valid JSON",
                status_code=response.status_code,
                response_body=raw_body,
                content_type=content_type,
            ) from exc

        try:
            envelope = LlamaChatCompletionResponse.model_validate(envelope_data)
        except ValidationError as exc:
            raise _CompletionFailure(
                "invalid-completion-envelope",
                f"invalid completion envelope: {exc}",
                status_code=response.status_code,
                response_body=raw_body,
                content_type=content_type,
            ) from exc

        return envelope, response.status_code, raw_body

    def _parse_assistant_content(
        self,
        envelope: LlamaChatCompletionResponse,
        *,
        response_model: type[ResponseT],
    ) -> ResponseT:
        if not envelope.choices:
            raise _CompletionFailure("empty-response", "missing choices")
        choice = envelope.choices[0]
        if choice.finish_reason == "length":
            raise _CompletionFailure("output-token-limit", "finish_reason=length")
        content = choice.message.content
        if content is None or content == "":
            raise _CompletionFailure("empty-response", "completion content was empty")
        try:
            return response_model.model_validate_json(content)
        except json.JSONDecodeError as exc:
            raise _CompletionFailure("invalid-json", "assistant content is not JSON") from exc
        except ValidationError as exc:
            raise _CompletionFailure(
                "invalid-structured-response",
                f"assistant content failed schema validation: {exc}",
            ) from exc

    @staticmethod
    def _may_retry(
        error_code: str,
        *,
        attempt_number: int,
        max_attempts: int,
        server_error_retries_used: int,
    ) -> bool:
        if attempt_number >= max_attempts:
            return False
        if error_code in {
            "response-body-truncated",
            "transport-timeout",
            "transport-error",
            "http-retryable",
        }:
            return True
        if error_code == "http-server-error":
            return server_error_retries_used < 1
        return False

    def _build_chat_payload(
        self,
        *,
        prompt: str,
        image_path: Path | None,
        response_format: dict[str, object] | None,
        thinking_contract: ThinkingControlContract,
        include_image: bool,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if include_image:
            if image_path is None:
                raise ValueError("image_path required when include_image=True")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_data_uri(image_path)},
                }
            )

        payload: dict[str, Any] = {
            "model": self._config.extraction.model_alias,
            "temperature": self._config.extraction.temperature,
            "seed": self._config.extraction.seed,
            "max_tokens": self._config.extraction.max_tokens,
            "messages": [{"role": "user", "content": content}],
            "chat_template_kwargs": {"enable_thinking": thinking_contract.enable_thinking},
            "reasoning_format": thinking_contract.reasoning_format,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        return payload

    def _image_data_uri(self, image_path: Path) -> str:
        image_bytes = image_path.read_bytes()
        encoded = base64.standard_b64encode(image_bytes).decode("ascii")
        suffix = image_path.suffix.lower()
        media_type = "image/png" if suffix == ".png" else "image/jpeg"
        return f"data:{media_type};base64,{encoded}"

    def _enforce_non_thinking_response(
        self,
        response: LlamaChatCompletionResponse,
        contract: ThinkingControlContract,
        *,
        smoke: bool = False,
    ) -> None:
        if not response.choices:
            if smoke:
                raise ProcessingError(code="thinking-control-contract-drift")
            raise _CompletionFailure("empty-response", "missing choices")
        message = response.choices[0].message
        if message.reasoning_content:
            if smoke:
                raise ProcessingError(code="thinking-control-contract-drift")
            raise _CompletionFailure(
                "unexpected-reasoning-content",
                "completion contained reasoning content",
            )
        if contract.reasoning_content_expected and not message.reasoning_content:
            raise ProcessingError(code="thinking-control-contract-drift")

    def _estimate_tokens(self, payload: dict[str, Any]) -> int:
        messages = payload.get("messages", [])
        text_parts: list[str] = []
        for message in cast(list[dict[str, Any]], messages):
            content = message.get("content")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for part in cast(list[dict[str, Any]], content):
                    if part.get("type") == "text":
                        text_parts.append(str(part.get("text", "")))
        prompt_text = "\n".join(text_parts)
        return max(1, len(prompt_text) // _ESTIMATE_CHARS_PER_TOKEN)

    def _effective_image_tokens(self) -> int:
        extraction = self._config.extraction
        if extraction.server_image_max_tokens is not None:
            return extraction.server_image_max_tokens
        return extraction.reserved_image_tokens

    def _build_request_summary_base(self, payload: dict[str, Any]) -> dict[str, object]:
        schema = load_wire_schema(self._config.extraction.wire_schema_version)
        normalized_schema = normalize_llama_schema(schema)
        return {
            "model": payload.get("model"),
            "max_tokens": payload.get("max_tokens"),
            "response_format_type": payload.get("response_format", {}).get("type"),
            "schema_sha256": sha256_hex(serialize_wire_request(normalized_schema)),
            "include_image": any(
                part.get("type") == "image_url"
                for message in cast(list[dict[str, Any]], payload.get("messages", []))
                for part in (
                    message.get("content", [])
                    if isinstance(message.get("content"), list)
                    else []
                )
            ),
        }

    def _failure_context(
        self,
        *,
        prompt: str,
        request_summary: dict[str, object],
        schema_ref: bytes,
        page_image_sha256: str,
        wire_request_sha256: str | None,
    ) -> InferenceFailureContext:
        return InferenceFailureContext(
            prompt=prompt.encode("utf-8"),
            request_summary=json.dumps(
                request_summary,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
            schema_ref=schema_ref,
            page_image_sha256=page_image_sha256,
            wire_request_sha256=wire_request_sha256,
        )

    def _empty_failure_context(self) -> InferenceFailureContext:
        return InferenceFailureContext(
            prompt=b"",
            request_summary=b"{}",
            schema_ref=b"{}",
            page_image_sha256="",
            wire_request_sha256=None,
        )

    def _sleep_backoff(self, attempt_number: int, retry_after_seconds: float | None) -> None:
        if retry_after_seconds is not None:
            time.sleep(
                min(retry_after_seconds, self._config.process.retry_after_max_seconds)
            )
            return
        delay = self._config.process.retry_backoff_seconds * (2 ** (attempt_number - 1))
        time.sleep(delay)

    def _parse_retry_after(self, header_value: str | None) -> float | None:
        if header_value is None:
            return None
        try:
            return float(header_value)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(header_value)
                return max(0.0, retry_at.timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                return None

    def _get_bytes(self, path: str) -> bytes:
        try:
            response = self._client.get(path)
        except httpx.TimeoutException as exc:
            raise ProcessingError(
                code="unsupported-llama-server-contract",
                message=f"timeout calling {path}",
            ) from exc
        except httpx.TransportError as exc:
            raise ProcessingError(
                code="unsupported-llama-server-contract",
                message=f"transport error calling {path}: {exc}",
            ) from exc
        if response.status_code != 200:
            raise ProcessingError(
                code="unsupported-llama-server-contract",
                message=f"{path} returned HTTP {response.status_code}",
            )
        return response.content

    def _context_size_from_props(self, props: LlamaPropsResponse) -> int:
        settings = props.default_generation_settings or {}
        for key in ("n_ctx", "context_size", "slot_prompt_capacity"):
            value = settings.get(key)
            if isinstance(value, int) and value > 0:
                return value
        raise ProcessingError(
            code="unsupported-llama-server-contract",
            message="could not determine context size from /props",
        )

    def _llama_cpp_build_from_props(self, props: LlamaPropsResponse) -> str | None:
        if props.build_info and props.build_info.strip():
            return props.build_info.strip()
        if props.build_commit:
            return props.build_commit
        if props.build_number is not None:
            return str(props.build_number)
        return None

    def _require_vision_modalities(self, props: LlamaPropsResponse) -> None:
        modalities = props.modalities
        if modalities is None:
            raise ProcessingError(
                code="unsupported-llama-server-contract",
                message="missing modalities in /props",
            )
        if "vision" not in modalities:
            raise ProcessingError(
                code="unsupported-llama-server-contract",
                message="missing modalities.vision in /props",
            )
        vision = modalities["vision"]
        if not isinstance(vision, bool):
            raise ProcessingError(
                code="unsupported-llama-server-contract",
                message="modalities.vision must be a boolean",
            )
        if not vision:
            raise ProcessingError(
                code="unsupported-llama-server-contract",
                message="modalities.vision is false",
            )

    def _apply_template_tokenize_supported(
        self,
        preflight: PreflightResult,
        *,
        text_payload: dict[str, Any],
    ) -> bool:
        messages = self._apply_template_projection(text_payload)
        caps = preflight.capabilities.chat_template_caps
        if self._template_supports_thinking(caps):
            false_prompt = self._request_apply_template(
                {
                    "messages": messages,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                discovery=True,
                extended=True,
            )
            true_prompt = self._request_apply_template(
                {
                    "messages": messages,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
                discovery=True,
                extended=True,
            )
            if false_prompt is None or true_prompt is None:
                return False
            if false_prompt == true_prompt:
                return False
            count = self._request_tokenize(
                false_prompt,
                add_special=False,
                parse_special=True,
                discovery=True,
            )
            return count is not None

        prompt = self._request_apply_template(
            {"messages": messages},
            discovery=True,
            extended=False,
        )
        if prompt is None:
            return False
        return self._request_tokenize(
            prompt,
            add_special=False,
            parse_special=True,
            discovery=True,
        ) is not None

    def _build_apply_template_contract(
        self,
        preflight: PreflightResult,
    ) -> ApplyTemplateTokenizeContract:
        identity = preflight.identity
        caps = preflight.capabilities.chat_template_caps
        request_mode: Literal["messages-only", "messages-plus-chat-template-kwargs"]
        if self._template_supports_thinking(caps):
            request_mode = "messages-plus-chat-template-kwargs"
        else:
            request_mode = "messages-only"
        return ApplyTemplateTokenizeContract(
            mode="apply-template-tokenize",
            apply_template_request_mode=request_mode,
            input_projection="text-only",
            image_token_policy="configured-reserve",
            model_alias=identity.model_alias,
            llama_cpp_build=identity.llama_cpp_build,
            chat_template_sha256=identity.chat_template_sha256,
        )

    @staticmethod
    def _template_supports_thinking(caps: dict[str, object]) -> bool:
        return caps.get("reasoning") is True

    def _apply_template_projection(
        self,
        text_payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return self._project_text_only_messages(text_payload)

    def _request_apply_template(
        self,
        body: dict[str, Any],
        *,
        discovery: bool,
        extended: bool,
    ) -> str | None:
        wire_body = serialize_wire_request(body)
        if b"image_url" in wire_body or b"data:" in wire_body:
            raise ProcessingError(
                code="token-counting-calibration-failed",
                message="apply-template projection must be text-only",
            )

        try:
            response = self._client.post(
                "/apply-template",
                content=wire_body,
                headers={"Content-Type": "application/json"},
            )
        except httpx.TimeoutException as exc:
            self._raise_counting_probe_transient(discovery, from_exc=exc)
        except httpx.TransportError as exc:
            self._raise_counting_probe_transient(discovery, from_exc=exc)

        if response.status_code == 404:
            return None
        if response.status_code in _TRANSIENT_STATUS_CODES or response.status_code >= 500:
            self._raise_counting_probe_transient(discovery)
        if response.status_code == 400:
            if discovery and extended:
                return None
            if discovery:
                raise ProcessingError(
                    code="token-counting-calibration-failed",
                    message="apply-template rejected documented request",
                )
            raise ProcessingError(code="token-counting-contract-drift")
        if response.status_code != 200:
            if discovery and extended:
                return None
            if discovery:
                raise ProcessingError(
                    code="token-counting-calibration-failed",
                    message=f"apply-template returned HTTP {response.status_code}",
                )
            raise ProcessingError(code="token-counting-contract-drift")

        try:
            parsed = json.loads(response.content)
        except json.JSONDecodeError:
            if discovery:
                raise ProcessingError(
                    code="token-counting-calibration-failed",
                    message="apply-template returned malformed JSON",
                ) from None
            raise ProcessingError(code="token-counting-contract-drift") from None

        if isinstance(parsed, dict):
            prompt = parsed.get("prompt")
            if isinstance(prompt, str) and prompt:
                return prompt
        if discovery:
            raise ProcessingError(
                code="token-counting-calibration-failed",
                message="apply-template returned malformed successful response",
            )
        raise ProcessingError(code="token-counting-contract-drift")

    def _request_tokenize(
        self,
        prompt: str,
        *,
        add_special: bool,
        parse_special: bool,
        discovery: bool,
    ) -> int | None:
        body = {
            "content": prompt,
            "add_special": add_special,
            "parse_special": parse_special,
        }
        try:
            response = self._client.post(
                "/tokenize",
                content=serialize_wire_request(body),
                headers={"Content-Type": "application/json"},
            )
        except httpx.TimeoutException as exc:
            self._raise_counting_probe_transient(discovery, from_exc=exc)
        except httpx.TransportError as exc:
            self._raise_counting_probe_transient(discovery, from_exc=exc)

        if response.status_code == 404:
            return None
        if response.status_code in _TRANSIENT_STATUS_CODES or response.status_code >= 500:
            self._raise_counting_probe_transient(discovery)
        if response.status_code == 400:
            if discovery:
                raise ProcessingError(
                    code="token-counting-calibration-failed",
                    message="tokenize rejected documented request",
                )
            raise ProcessingError(code="token-counting-contract-drift")
        if response.status_code != 200:
            if discovery:
                raise ProcessingError(
                    code="token-counting-calibration-failed",
                    message=f"tokenize returned HTTP {response.status_code}",
                )
            raise ProcessingError(code="token-counting-contract-drift")

        try:
            parsed = json.loads(response.content)
        except json.JSONDecodeError:
            if discovery:
                raise ProcessingError(
                    code="token-counting-calibration-failed",
                    message="tokenize returned malformed JSON",
                ) from None
            raise ProcessingError(code="token-counting-contract-drift") from None

        tokens = parsed.get("tokens") if isinstance(parsed, dict) else None
        if not isinstance(tokens, list):
            if discovery:
                raise ProcessingError(
                    code="token-counting-calibration-failed",
                    message="tokenize missing tokens array",
                )
            raise ProcessingError(code="token-counting-contract-drift")
        return len(tokens)

    def _media_marker(self, props: LlamaPropsResponse) -> str | None:
        if props.media_marker is not None:
            return str(props.media_marker)
        modalities = props.modalities or {}
        marker = modalities.get("media_marker")
        return str(marker) if marker is not None else None

    def _chat_template_caps(self, props: LlamaPropsResponse) -> dict[str, object]:
        if isinstance(props.chat_template_caps, dict):
            return dict(props.chat_template_caps)
        settings = props.default_generation_settings or {}
        caps = settings.get("chat_template_caps")
        if isinstance(caps, dict):
            return caps
        return {}

    def _applied_template_probe_supported(
        self,
        capabilities: ServerInvocationCapabilities,
    ) -> bool:
        caps = capabilities.chat_template_caps
        return bool(caps.get("applied_template_probe") or caps.get("apply_template"))


class _ThinkingCandidateRejected(Exception):
    """Deterministic rejection of a reasoning-format calibration candidate."""


class _ProbeTransient(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _TransportFailure(Exception):
    def __init__(
        self,
        *,
        error_code: str,
        message: str,
        status_code: int | None = None,
        body: bytes | None = None,
        content_type: str | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.body = body
        self.content_type = content_type
        self.retry_after_seconds = retry_after_seconds


class _CompletionFailure(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int | None = None,
        response_body: bytes | None = None,
        content_type: str | None = None,
    ) -> None:
        self.code = code
        self.status_code = status_code
        self.response_body = response_body
        self.content_type = content_type
        super().__init__(message)
