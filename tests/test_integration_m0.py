"""Optional live llama-server integration tests."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("BOOKEXTRACT_LLAMA_URL") is None,
    reason="set BOOKEXTRACT_LLAMA_URL to run live M0 contract tests",
)
def test_live_preflight() -> None:
    from bookextract.config import ExtractionConfig, ProcessingConfig, ProcessOptions
    from bookextract.inference.llamacpp import LlamaCppVisionClient

    config = ProcessingConfig(
        extraction=ExtractionConfig(
            model_alias=os.environ.get("BOOKEXTRACT_MODEL_ALIAS", "default"),
            prompt_version="v1",
        ),
        process=ProcessOptions(llama_base_url=os.environ["BOOKEXTRACT_LLAMA_URL"]),
    )
    with LlamaCppVisionClient(config) as client:
        preflight = client.preflight()
        assert preflight.identity.context_size > 0
