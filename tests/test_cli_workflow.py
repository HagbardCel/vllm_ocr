"""CLI workflow, init guards, relocation, and calibration lifecycle tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from bookextract.canonical import sha256_hex
from bookextract.cli import main
from bookextract.config import (
    ExtractionConfig,
    InferenceLocation,
    ProcessingConfig,
    ProcessOptions,
    RenderContract,
    RunRecord,
    write_json_atomic,
)
from bookextract.context import build_page_context
from bookextract.errors import ProcessingError
from bookextract.fingerprints import fingerprint_file
from bookextract.inference.bootstrap import prepare_inference_environment
from bookextract.inference.llamacpp import LlamaCppVisionClient
from bookextract.models import InferenceEnvironment, ServerInferenceIdentity
from bookextract.storage import RunStore
from tests.conftest import make_inference_environment


def _write_config(path: Path, *, llama_base_url: str = "http://127.0.0.1:8080") -> None:
    path.write_text(
        f"""
[extraction]
model_alias = "vision-model"
prompt_version = "v1"

[process]
llama_base_url = "{llama_base_url}"
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _run_artifacts(run_dir: Path) -> set[str]:
    names: set[str] = set()
    for path in (
        run_dir / "run.json",
        run_dir / "head.json",
        run_dir / "inference-location.json",
    ):
        if path.exists():
            names.add(path.name)
    return names


def test_init_requires_model_arg(minimal_pdf: Path, tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    run_dir = tmp_path / "run"
    _write_config(config)
    with pytest.raises(SystemExit):
        main(
            [
                "init",
                str(minimal_pdf),
                "--run",
                str(run_dir),
                "--config",
                str(config),
            ]
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["init", "missing.pdf", "--run", "r", "--config", "c", "--model", "m"],
        ["init", "pdf", "--run", "r", "--config", "missing.toml", "--model", "m"],
        ["init", "pdf", "--run", "r", "--config", "c", "--model", "missing.gguf"],
        [
            "init",
            "pdf",
            "--run",
            "r",
            "--config",
            "c",
            "--model",
            "m",
            "--projector",
            "missing.proj",
        ],
    ],
)
def test_init_pre_write_failures_leave_no_run_artifacts(
    minimal_pdf: Path,
    tmp_path: Path,
    argv: list[str],
) -> None:
    config = tmp_path / "config.toml"
    model = tmp_path / "model.gguf"
    projector = tmp_path / "projector.gguf"
    run_dir = tmp_path / "run"
    _write_config(config)
    model.write_bytes(b"model")
    projector.write_bytes(b"proj")

    replacements = {
        "pdf": str(minimal_pdf),
        "missing.pdf": str(tmp_path / "missing.pdf"),
        "c": str(config),
        "missing.toml": str(tmp_path / "missing.toml"),
        "m": str(model),
        "missing.gguf": str(tmp_path / "missing.gguf"),
        "missing.proj": str(tmp_path / "missing.proj"),
        "r": str(run_dir),
    }
    resolved = [replacements.get(token, token) for token in argv]
    exit_code = main(resolved)
    assert exit_code == 2
    assert _run_artifacts(run_dir) == set()


def test_init_invalid_config_leaves_no_run_artifacts(
    minimal_pdf: Path, tmp_path: Path
) -> None:
    config = tmp_path / "config.toml"
    model = tmp_path / "model.gguf"
    run_dir = tmp_path / "run"
    config.write_text("not-valid-toml[[[\n", encoding="utf-8")
    model.write_bytes(b"model")
    exit_code = main(
        [
            "init",
            str(minimal_pdf),
            "--run",
            str(run_dir),
            "--config",
            str(config),
            "--model",
            str(model),
        ]
    )
    assert exit_code != 0
    assert _run_artifacts(run_dir) == set()


def test_init_persists_process_options(minimal_pdf: Path, tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    model = tmp_path / "model.gguf"
    run_dir = tmp_path / "run"
    _write_config(config, llama_base_url="http://custom:9999")
    model.write_bytes(b"model")
    assert (
        main(
            [
                "init",
                str(minimal_pdf),
                "--run",
                str(run_dir),
                "--config",
                str(config),
                "--model",
                str(model),
            ]
        )
        == 0
    )
    record = RunRecord.model_validate_json((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record.process_options.llama_base_url == "http://custom:9999"


def _initialized_run(
    tmp_path: Path,
    minimal_pdf: Path,
    *,
    model_bytes: bytes = b"model-a",
    projector_bytes: bytes | None = None,
) -> tuple[Path, Path, Path | None]:
    config = tmp_path / "config.toml"
    model = tmp_path / "model.gguf"
    projector = tmp_path / "projector.gguf" if projector_bytes is not None else None
    run_dir = tmp_path / "run"
    _write_config(config)
    model.write_bytes(model_bytes)
    argv = [
        "init",
        str(minimal_pdf),
        "--run",
        str(run_dir),
        "--config",
        str(config),
        "--model",
        str(model),
    ]
    if projector is not None:
        projector.write_bytes(projector_bytes or b"proj")
        argv.extend(["--projector", str(projector)])
    assert main(argv) == 0
    return run_dir, model, projector


def test_relocate_rejected_leaves_location_unchanged(
    tmp_path: Path, minimal_pdf: Path
) -> None:
    run_dir, model, _ = _initialized_run(tmp_path, minimal_pdf)
    store = RunStore(run_dir)
    env = make_inference_environment().model_copy(
        update={
            "fingerprints_complete": True,
            "model_file": fingerprint_file(model),
        }
    )
    store.write_inference_environment(env)
    location_path = run_dir / "inference-location.json"
    original = location_path.read_bytes()
    bad_model = tmp_path / "other.gguf"
    bad_model.write_bytes(b"different-model")
    exit_code = main(
        [
            "relocate-inference-files",
            "--run",
            str(run_dir),
            "--model",
            str(bad_model),
        ]
    )
    assert exit_code == 13
    assert location_path.read_bytes() == original


def test_relocate_accepts_equal_sha_different_path(
    tmp_path: Path, minimal_pdf: Path
) -> None:
    run_dir, model, _ = _initialized_run(tmp_path, minimal_pdf, model_bytes=b"same-content")
    store = RunStore(run_dir)
    model_fp = fingerprint_file(model)
    env = make_inference_environment().model_copy(
        update={
            "fingerprints_complete": True,
            "model_file": model_fp,
            "server": make_inference_environment().server.model_copy(
                update={"server_reported_model_path": str(model)}
            ),
        }
    )
    store.write_inference_environment(env)
    new_model = tmp_path / "alias.gguf"
    new_model.write_bytes(b"same-content")
    assert (
        main(
            [
                "relocate-inference-files",
                "--run",
                str(run_dir),
                "--model",
                str(new_model),
            ]
        )
        == 0
    )
    location = store.load_inference_location()
    assert location.model_file_path == new_model.resolve()


def test_relocate_projector_only_verifies_retained_model(
    tmp_path: Path, minimal_pdf: Path
) -> None:
    run_dir, model, projector = _initialized_run(
        tmp_path,
        minimal_pdf,
        model_bytes=b"model",
        projector_bytes=b"proj-a",
    )
    store = RunStore(run_dir)
    model_fp = fingerprint_file(model)
    projector_fp = fingerprint_file(projector)
    env = make_inference_environment().model_copy(
        update={
            "fingerprints_complete": True,
            "model_file": model_fp,
            "projector_file": projector_fp,
            "server": make_inference_environment().server.model_copy(
                update={"server_reported_model_path": str(model)}
            ),
        }
    )
    store.write_inference_environment(env)
    new_projector = tmp_path / "new.proj"
    new_projector.write_bytes(b"proj-a")
    assert (
        main(
            [
                "relocate-inference-files",
                "--run",
                str(run_dir),
                "--projector",
                str(new_projector),
            ]
        )
        == 0
    )
    location = store.load_inference_location()
    assert location.model_file_path == model.resolve()
    assert location.projector_file_path == new_projector.resolve()


def test_relocate_incomplete_fingerprints_fail_closed(
    tmp_path: Path, minimal_pdf: Path
) -> None:
    run_dir, model, _ = _initialized_run(tmp_path, minimal_pdf)
    store = RunStore(run_dir)
    env = make_inference_environment().model_copy(update={"fingerprints_complete": False})
    store.write_inference_environment(env)
    alias = tmp_path / "alias.gguf"
    alias.write_bytes(b"model-a")
    exit_code = main(
        [
            "relocate-inference-files",
            "--run",
            str(run_dir),
            "--model",
            str(alias),
        ]
    )
    assert exit_code == 13


def test_first_process_passes_render_callback(
    tmp_path: Path, minimal_pdf: Path
) -> None:
    run_dir, _, _ = _initialized_run(tmp_path, minimal_pdf)
    captured: dict[str, object] = {}

    def fake_prepare(**kwargs: object) -> InferenceEnvironment:
        captured["callback"] = kwargs.get("render_calibration_page")
        return make_inference_environment()

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    with (
        patch(
            "bookextract.inference.bootstrap.prepare_inference_environment",
            side_effect=fake_prepare,
        ),
        patch("bookextract.inference.llamacpp.LlamaCppVisionClient", return_value=mock_client),
        patch("bookextract.cli.process_book"),
        patch("bookextract.interpretation.vlm.LlamaCppStructuredClient"),
        patch("bookextract.interpretation.vlm.VlmPageInterpreter"),
    ):
        assert main(["process", "--run", str(run_dir), "--interpreter", "vlm"]) == 0

    assert captured["callback"] is not None


def test_later_process_skips_render_callback(tmp_path: Path, minimal_pdf: Path) -> None:
    run_dir, _, _ = _initialized_run(tmp_path, minimal_pdf)
    store = RunStore(run_dir)
    store.write_inference_environment(make_inference_environment())
    captured: dict[str, object] = {}

    def fake_prepare(**kwargs: object) -> InferenceEnvironment:
        captured["callback"] = kwargs.get("render_calibration_page")
        return make_inference_environment()

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    with (
        patch(
            "bookextract.inference.bootstrap.prepare_inference_environment",
            side_effect=fake_prepare,
        ),
        patch("bookextract.inference.llamacpp.LlamaCppVisionClient", return_value=mock_client),
        patch("bookextract.cli.process_book"),
        patch("bookextract.interpretation.vlm.LlamaCppStructuredClient"),
        patch("bookextract.interpretation.vlm.VlmPageInterpreter"),
    ):
        assert main(["process", "--run", str(run_dir), "--interpreter", "vlm"]) == 0

    assert captured["callback"] is None


def test_calibration_render_failure_persisted(tmp_path: Path, minimal_pdf: Path) -> None:
    run_dir, _, _ = _initialized_run(tmp_path, minimal_pdf)
    store = RunStore(run_dir)

    def failing_render() -> None:
        raise ProcessingError(code="page-render-failed", message="render failed")

    config = ProcessingConfig(
        extraction=ExtractionConfig(model_alias="vision-model", prompt_version="v1"),
        process=ProcessOptions(),
    )
    mock_client = MagicMock()
    mock_client._config = config
    mock_client.preflight.return_value = MagicMock(identity=MagicMock())
    thinking_contract = make_inference_environment().thinking_control_contract
    mock_client.calibrate_thinking_control.return_value = thinking_contract

    from bookextract.interpretation.prompts import PagePromptBuilder
    from bookextract.state import load_or_initialize_state

    state, _ = load_or_initialize_state(store)
    context = build_page_context(state)
    prompt = PagePromptBuilder().build(context)
    with pytest.raises(ProcessingError) as exc_info:
        prepare_inference_environment(
            store=store,
            client=mock_client,
            calibration_context=context,
            calibration_prompt=prompt,
            render_calibration_page=failing_render,
        )
    assert exc_info.value.code == "page-render-failed"

    failure_dir = run_dir / "failures" / "page-0001"
    assert failure_dir.is_dir()
    assert store.read_head().committed_page_count == 0


def test_bootstrap_later_environment_never_calls_render(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    store.ensure_layout()
    write_json_atomic(
        tmp_path / "run" / "run.json",
        RunRecord(
            source={"sha256": "0" * 64, "size": 1},
            extraction=ExtractionConfig(model_alias="vision-model", prompt_version="v1"),
            fingerprint_policy={"require_complete_fingerprint": False},
            process_options=ProcessOptions(),
            render_contract=RenderContract(pymupdf_version="1"),
            prompt_sha256="0" * 64,
            wire_schema_sha256="0" * 64,
            created_at="2026-01-01T00:00:00Z",
        ).model_dump(mode="json"),
    )
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    write_json_atomic(
        tmp_path / "run" / "inference-location.json",
        InferenceLocation(model_file_path=model).model_dump(mode="json"),
    )
    base_env = make_inference_environment()
    store.write_inference_environment(
        base_env.model_copy(
            update={
                "fingerprints_complete": True,
                "server": ServerInferenceIdentity(
                    llama_cpp_build="b1",
                    model_alias="vision-model",
                    context_size=32768,
                    vision_supported=True,
                    chat_template_sha256=sha256_hex(b""),
                    server_reported_model_path=str(model),
                ),
                "model_file": fingerprint_file(model),
                "token_counting_contract": base_env.token_counting_contract.model_copy(
                    update={
                        "model_alias": "vision-model",
                        "llama_cpp_build": "b1",
                        "chat_template_sha256": sha256_hex(b""),
                    }
                ),
                "thinking_control_contract": base_env.thinking_control_contract.model_copy(
                    update={
                        "model_alias": "vision-model",
                        "llama_cpp_build": "b1",
                        "chat_template_sha256": sha256_hex(b""),
                    }
                ),
            }
        )
    )

    calls = {"count": 0}

    def render() -> None:
        calls["count"] += 1
        raise AssertionError("render should not run")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, content=b"ok")
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                content=json.dumps({"data": [{"id": "vision-model"}]}).encode(),
            )
        if request.url.path == "/props":
            return httpx.Response(
                200,
                content=json.dumps(
                    {
                        "build_info": "b1",
                        "model_path": str(model),
                        "modalities": {"vision": True},
                        "default_generation_settings": {"n_ctx": 32768},
                        "chat_template": "",
                    }
                ).encode(),
            )
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                content=json.dumps(
                    {
                        "id": "1",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "vision-model",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"content": '{"page_type":"blank","blocks":[]}'},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                ).encode(),
            )
        return httpx.Response(404)

    from bookextract.config import ProcessingConfig

    config = ProcessingConfig(
        extraction=ExtractionConfig(model_alias="vision-model", prompt_version="v1"),
        process=ProcessOptions(llama_base_url="http://test"),
    )
    with LlamaCppVisionClient(
        config,
        client=httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)),
    ) as client:
        prepare_inference_environment(
            store=store,
            client=client,
            calibration_context=MagicMock(),
            calibration_prompt="hello",
            render_calibration_page=render,
        )

    assert calls["count"] == 0
