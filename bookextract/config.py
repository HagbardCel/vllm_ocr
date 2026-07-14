"""Configuration models and loaders."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from bookextract.models import DomainModel


class ExtractionConfig(DomainModel):
    render_dpi: int = 240
    render_annotations: bool = False
    model_alias: str
    prompt_version: str
    wire_schema_version: str = "vlm-page-response-v1"
    temperature: float = 0.0
    seed: int = 0
    max_tokens: int = 8192
    require_figure_crops: bool = True
    require_same_page_footnotes: bool = True
    reserved_image_tokens: int = 2048
    context_safety_margin_tokens: int = 512
    server_image_max_tokens: int | None = None


class InferenceFingerprintConfig(DomainModel):
    require_complete_fingerprint: bool = False


class ProcessOptions(DomainModel):
    llama_base_url: str = "http://127.0.0.1:8080"
    request_timeout_seconds: float = 300
    max_attempts: int = 2
    retry_backoff_seconds: float = 1.0
    retry_after_max_seconds: float = 60.0


class MarkdownRenderConfig(DomainModel):
    include_page_markers: bool = True


class EpubRenderConfig(DomainModel):
    pandoc_executable: str = "pandoc"
    epubcheck_executable: str = "epubcheck"
    epubcheck_jar_path: Path | None = None
    include_toc: bool = True


class ProcessingConfig(DomainModel):
    extraction: ExtractionConfig
    fingerprint: InferenceFingerprintConfig = Field(default_factory=InferenceFingerprintConfig)
    process: ProcessOptions = Field(default_factory=ProcessOptions)
    markdown: MarkdownRenderConfig = Field(default_factory=MarkdownRenderConfig)
    epub: EpubRenderConfig = Field(default_factory=EpubRenderConfig)


class RenderContract(DomainModel):
    render_contract_format_version: Literal[1]
    colorspace: Literal["RGB"] = "RGB"
    alpha: Literal[False] = False
    pymupdf_version: str


class RunRecord(DomainModel):
    run_format_version: Literal[1]
    source: dict[str, object]
    extraction: ExtractionConfig
    fingerprint_policy: InferenceFingerprintConfig
    process_options: ProcessOptions
    markdown: MarkdownRenderConfig
    epub: EpubRenderConfig
    render_contract: RenderContract
    prompt_sha256: str
    wire_schema_sha256: str
    created_at: str


class SourceLocation(DomainModel):
    source_location_format_version: Literal[1]
    pdf_path: Path


class InferenceLocation(DomainModel):
    inference_location_format_version: Literal[1]
    model_file_path: Path
    projector_file_path: Path | None = None


def load_config_from_dict(data: dict[str, Any]) -> ProcessingConfig:
    return ProcessingConfig.model_validate(data)


def load_config_from_toml(path: Path) -> ProcessingConfig:
    with path.open("rb") as handle:
        return load_config_from_dict(tomllib.load(handle))


def load_processing_config(path: Path) -> ProcessingConfig:
    return load_config_from_toml(path)


def load_run_record(run_dir: Path) -> RunRecord:
    return RunRecord.model_validate_json(
        (run_dir / "run.json").read_text(encoding="utf-8")
    )


def processing_config_from_run_record(record: RunRecord) -> ProcessingConfig:
    return ProcessingConfig(
        extraction=record.extraction,
        fingerprint=record.fingerprint_policy,
        process=record.process_options,
        markdown=record.markdown,
        epub=record.epub,
    )


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
