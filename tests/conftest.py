"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from bookextract.config import (
    EpubRenderConfig,
    ExtractionConfig,
    MarkdownRenderConfig,
    ProcessingConfig,
    ProcessOptions,
)
from bookextract.models import (
    ApplyTemplateTokenizeContract,
    InferenceEnvironment,
    ModelRunInfo,
    PageAssessment,
    PageInterpretation,
    ServerInferenceIdentity,
    ThinkingControlContract,
)


@pytest.fixture
def extraction_config() -> ExtractionConfig:
    return ExtractionConfig(
        model_alias="test-model",
        prompt_version="v1",
        require_figure_crops=False,
        require_same_page_footnotes=False,
    )


@pytest.fixture
def processing_config(extraction_config: ExtractionConfig) -> ProcessingConfig:
    return ProcessingConfig(extraction=extraction_config)


def make_model_run() -> ModelRunInfo:
    return ModelRunInfo(backend="test", model="test-model", prompt_version="v1")


def make_assessment(
    page_index: int,
    interpretation: PageInterpretation,
    *,
    image_path: Path | None = None,
) -> PageAssessment:
    return PageAssessment(
        page_index=page_index,
        image_path=image_path or Path(f"pages/page-{page_index + 1:04d}.png"),
        interpretation=interpretation,
        model_run=make_model_run(),
    )


def make_inference_environment() -> InferenceEnvironment:
    return InferenceEnvironment(
        inference_environment_format_version=1,
        server=ServerInferenceIdentity(
            llama_cpp_build="test-build",
            model_alias="test-model",
            context_size=32768,
            vision_supported=True,
            chat_template_sha256="c" * 64,
            server_reported_model_path="/models/test.gguf",
        ),
        model_binding_verified=True,
        projector_binding="operator-asserted",
        token_counting_contract=ApplyTemplateTokenizeContract(
            contract_format_version=1,
            mode="apply-template-tokenize",
            apply_template_request_mode="messages-only",
            input_projection="text-only",
            image_token_policy="configured-reserve",
            model_alias="test-model",
            llama_cpp_build="test-build",
            chat_template_sha256="c" * 64,
        ),
        thinking_control_contract=ThinkingControlContract(
            contract_format_version=1,
            reasoning_format="none",
            applied_template_probe_supported=False,
            model_alias="test-model",
            llama_cpp_build="test-build",
            chat_template_sha256="c" * 64,
        ),
    )


@pytest.fixture
def minimal_pdf(tmp_path: Path) -> Path:
    """Create a one-page blank PDF for layout tests."""
    pdf_path = tmp_path / "minimal.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Test page")
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def multi_page_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "multi.pdf"
    doc = fitz.open()
    for _ in range(3):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), "Test page")
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def run_dir(tmp_path: Path, minimal_pdf: Path) -> Path:
    """Initialize a minimal run directory layout."""
    from bookextract.config import RenderContract, RunRecord, SourceLocation, write_json_atomic
    from bookextract.storage import RunStore

    run = tmp_path / "run"
    store = RunStore(run)
    store.ensure_layout()

    write_json_atomic(
        run / "run.json",
        RunRecord(
            run_format_version=1,
            source={"sha256": "0" * 64, "size": 1},
            extraction=ExtractionConfig(model_alias="test-model", prompt_version="v1"),
            fingerprint_policy={"require_complete_fingerprint": False},
            process_options=ProcessOptions(),
            markdown=MarkdownRenderConfig(),
            epub=EpubRenderConfig(),
            render_contract=RenderContract(
                render_contract_format_version=1,
                pymupdf_version=fitz.__version__,
            ),
            prompt_sha256="0" * 64,
            wire_schema_sha256="0" * 64,
            created_at="2026-01-01T00:00:00Z",
        ).model_dump(mode="json"),
    )
    write_json_atomic(
        run / "source-location.json",
        SourceLocation(source_location_format_version=1, pdf_path=minimal_pdf).model_dump(
            mode="json"
        ),
    )
    store.write_head(0)
    return run
