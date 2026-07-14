"""M0 contract model and static feasibility tests."""

from __future__ import annotations

import pytest

from bookextract.config import ExtractionConfig, ProcessingConfig
from bookextract.errors import ProcessingError
from bookextract.inference.llamacpp import LlamaCppVisionClient
from bookextract.models import (
    ContextBudgetResult,
    InferenceEnvironment,
    MultimodalInputTokensContract,
)
from tests.conftest import make_inference_environment


def test_token_counting_contract_discriminator() -> None:
    contract = MultimodalInputTokensContract(
        contract_format_version=1,
        mode="chat-input-tokens-multimodal",
        model_alias="m",
        llama_cpp_build="b",
        chat_template_sha256="c",
    )
    env = make_inference_environment().model_copy(
        update={"token_counting_contract": contract}
    )
    restored = InferenceEnvironment.model_validate(env.model_dump(mode="json"))
    assert restored.token_counting_contract.mode == "chat-input-tokens-multimodal"


def test_context_budget_required_tokens() -> None:
    budget = ContextBudgetResult(
        counted_input_tokens=100,
        image_tokens_reserved=2048,
        output_tokens_reserved=8192,
        safety_margin_tokens=512,
        context_size=32768,
        counting_mode="apply-template-tokenize",
        exact_for_projected_input=True,
        multimodal_count_included=False,
    )
    assert budget.required_tokens == 100 + 2048 + 8192 + 512


def test_static_context_impossible() -> None:
    config = ProcessingConfig(
        extraction=ExtractionConfig(
            model_alias="m",
            prompt_version="v1",
            max_tokens=8000,
            context_safety_margin_tokens=1000,
        )
    )
    client = LlamaCppVisionClient(config)
    with pytest.raises(ProcessingError, match="static-context-impossible"):
        client.check_static_context_feasibility(8000)
