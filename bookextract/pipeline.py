"""Parser-independent sequential book processing loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from bookextract.artifacts import (
    InferenceAttempt,
    InterpretationResult,
    validate_artifact_filenames,
)
from bookextract.config import ProcessingConfig
from bookextract.context import build_page_context
from bookextract.errors import InferenceError, ProcessingError, StructuralError
from bookextract.models import (
    BookDocument,
    BookState,
    ContentBlock,
    FigureBlock,
    PageAssessment,
    PageContext,
    PageInput,
    PageInterpretation,
    RenderedPage,
)
from bookextract.state import (
    apply_assessment,
    assign_block_ids,
    build_commit_payload,
    finalize_toc_if_required,
    load_or_initialize_state,
)
from bookextract.storage import RunStore
from bookextract.validation import validate_assessment, validate_complete_book


def _collect_figure_assets(
    interpretation: PageInterpretation,
    rendered: RenderedPage,
) -> tuple[PageInterpretation, dict[str, bytes]]:
    from bookextract.assets import crop_figure_asset

    extra_files: dict[str, bytes] = {}
    updated_blocks: list[ContentBlock] = []
    for block in interpretation.blocks:
        if block.kind != "figure":
            updated_blocks.append(block)
            continue
        figure: FigureBlock = block
        if figure.bbox is None:
            updated_blocks.append(block)
            continue
        block_suffix = figure.block_id or "figure"
        del block_suffix
        png_bytes, asset_sha256 = crop_figure_asset(rendered, figure.bbox)
        from bookextract.output_paths import figure_asset_path

        rel_path = figure_asset_path(asset_sha256)
        extra_files[rel_path] = png_bytes
        updated_blocks.append(figure.model_copy(update={"asset_sha256": asset_sha256}))
    return interpretation.model_copy(update={"blocks": updated_blocks}), extra_files


class PageSource(Protocol):
    @property
    def page_count(self) -> int: ...

    def render_page(self, page_index: int, *, dpi: int) -> RenderedPage: ...


class PageInterpreter(Protocol):
    def interpret(
        self,
        *,
        page_input: PageInput,
        context: PageContext,
    ) -> InterpretationResult: ...


@dataclass
class ProcessResult:
    document: BookDocument
    state: BookState


def _assessment_from_result(
    *,
    page_input: PageInput,
    result: InterpretationResult,
) -> PageAssessment:
    from bookextract.models import ModelRunInfo

    provenance = result.provenance
    return PageAssessment(
        page_index=page_input.page_index,
        image_path=page_input.image_path,
        interpretation=result.interpretation,
        model_run=ModelRunInfo(
            backend=provenance.backend,
            model=provenance.model,
            prompt_version=provenance.prompt_version,
            prompt_tokens=provenance.prompt_tokens,
            completion_tokens=provenance.completion_tokens,
            elapsed_ms=provenance.elapsed_ms,
        ),
    )


def process_book(
    *,
    source: PageSource,
    interpreter: PageInterpreter,
    store: RunStore,
    config: ProcessingConfig,
    through_page: int | None = None,
    max_pages: int | None = None,
) -> ProcessResult:
    """Process pages sequentially; never imports wire or conversion modules."""
    store.recover()
    state, committed = load_or_initialize_state(store)

    document = BookDocument()
    if committed:
        from bookextract.state import load_book_state_from_commits

        state = load_book_state_from_commits(store, committed)
        for page_number in range(1, committed + 1):
            raw = store.read_commit_file(page_number, "page-assessment.json")
            document.pages.append(
                PageAssessment.model_validate_json(raw.decode("utf-8"))
            )

    start_index = state.processed_page_count
    end_index = source.page_count
    if through_page is not None:
        end_index = min(end_index, through_page)
    if max_pages is not None:
        end_index = min(end_index, start_index + max_pages)

    for page_index in range(start_index, end_index):
        context = build_page_context(state)
        page_input: PageInput | None = None
        is_final_source_page = page_index + 1 == source.page_count

        try:
            rendered = source.render_page(
                page_index, dpi=config.extraction.render_dpi
            )
            page_input = PageInput(
                page_index=page_index,
                rendered=rendered,
            )
            result = interpreter.interpret(page_input=page_input, context=context)
        except ProcessingError as exc:
            if exc.code not in {"page-image-too-large", "page-render-failed"}:
                raise
            from bookextract.failure_persistence import persist_page_preparation_failure

            persist_page_preparation_failure(
                store=store,
                page_index=page_index,
                context=context.model_dump(mode="json"),
                extraction_config=config.extraction,
                error=exc,
            )
            raise
        except InferenceError as exc:
            planned_input = {
                "page_index": page_index,
                "stage": "rendering" if page_input is None else "interpretation",
                "render_dpi": config.extraction.render_dpi,
                "render_annotations": config.extraction.render_annotations,
            }
            store.persist_failure(
                page_number=page_index + 1,
                context=context.model_dump(mode="json"),
                page_input=(
                    page_input.model_dump(mode="json")
                    if page_input is not None
                    else planned_input
                ),
                prompt=exc.context.prompt,
                schema_ref={"wire_schema_version": config.extraction.wire_schema_version},
                request_summary=json.loads(exc.context.request_summary.decode("utf-8")),
                error={"code": exc.code, "message": str(exc)},
                attempts=None
                if not exc.attempts
                else {
                    f"attempt-{attempt.attempt_number:02d}/response.bin": attempt.response_body
                    or b""
                    for attempt in exc.attempts
                    if isinstance(attempt, InferenceAttempt)
                },
            )
            raise

        interpretation = assign_block_ids(result.interpretation, page_index)
        validate_artifact_filenames(result.artifacts)
        result = InterpretationResult(
            interpretation=interpretation,
            provenance=result.provenance,
            artifacts=result.artifacts,
            failed_attempts=result.failed_attempts,
        )

        assessment = _assessment_from_result(page_input=page_input, result=result)
        state = finalize_toc_if_required(state, assessment)

        try:
            validate_assessment(
                assessment=assessment,
                state=state,
                config=config.extraction,
            )
        except StructuralError as exc:
            store.persist_failure(
                page_number=page_index + 1,
                context=context.model_dump(mode="json"),
                page_input=page_input.model_dump(mode="json"),
                prompt=b"",
                schema_ref={"wire_schema_version": config.extraction.wire_schema_version},
                request_summary={},
                error={"code": exc.code, "message": str(exc)},
            )
            raise

        interpretation, figure_files = _collect_figure_assets(interpretation, rendered)
        assessment = assessment.model_copy(update={"interpretation": interpretation})

        if is_final_source_page:
            prospective_state = apply_assessment(state, assessment)
            prospective_document = BookDocument(pages=[*document.pages, assessment])
            try:
                validate_complete_book(
                    document=prospective_document,
                    state=prospective_state,
                )
            except StructuralError as exc:
                store.persist_failure(
                    page_number=page_index + 1,
                    context=context.model_dump(mode="json"),
                    page_input=page_input.model_dump(mode="json"),
                    prompt=b"",
                    schema_ref={"wire_schema_version": config.extraction.wire_schema_version},
                    request_summary={},
                    error={"code": exc.code, "message": str(exc)},
                )
                raise

        state = apply_assessment(state, assessment)
        document.pages.append(assessment)

        extra_files: dict[str, bytes] = dict(figure_files)
        for artifact in result.artifacts:
            extra_files[artifact.filename] = artifact.content

        commit_files = build_commit_payload(
            assessment=assessment,
            context_json=context.model_dump(mode="json"),
            interpretation_json=interpretation.model_dump(mode="json"),
            provenance_json=result.provenance.model_dump(mode="json"),
            extra_files=extra_files or None,
        )
        page_number = page_index + 1
        store.write_commit(page_number, commit_files)
        store.write_state_cache(state, page_number)

    return ProcessResult(document=document, state=state)
