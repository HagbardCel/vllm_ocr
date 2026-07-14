"""Shared page-preparation failure persistence for bootstrap and pipeline."""

from __future__ import annotations

from typing import Any

from bookextract.config import ExtractionConfig
from bookextract.errors import ProcessingError
from bookextract.storage import RunStore


def persist_page_preparation_failure(
    *,
    store: RunStore,
    page_index: int,
    context: dict[str, Any],
    extraction_config: ExtractionConfig,
    error: ProcessingError,
) -> None:
    store.persist_failure(
        page_number=page_index + 1,
        context=context,
        page_input={
            "page_index": page_index,
            "stage": "rendering",
            "render_dpi": extraction_config.render_dpi,
            "render_annotations": extraction_config.render_annotations,
        },
        prompt=b"",
        schema_ref={},
        request_summary={},
        error={"code": error.code, "message": str(error)},
        attempts=None,
    )
