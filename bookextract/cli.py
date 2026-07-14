"""bookextract command-line interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tomllib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import fitz
from pydantic import ValidationError

from bookextract.config import (
    InferenceLocation,
    ProcessingConfig,
    RenderContract,
    RunRecord,
    SourceLocation,
    load_processing_config,
    load_run_record,
    processing_config_from_run_record,
    write_json_atomic,
)
from bookextract.errors import BookExtractError, ProcessingError, exit_code_for_error
from bookextract.fingerprints import fingerprint_file
from bookextract.interpretation.prompts import prompt_sha256
from bookextract.output_publish import (
    publish_output,
    quarantine_invalid_output_destinations,
    recover_output_transaction,
)
from bookextract.pdf import PdfPageSource
from bookextract.pipeline import process_book
from bookextract.rendering.epub import EpubRenderer
from bookextract.rendering.markdown import MarkdownRenderer
from bookextract.rendering.output_bundle import (
    build_output_manifest,
    collect_commit_assets,
    write_output_bundle,
)
from bookextract.rendering.publication import build_publication_document
from bookextract.run_guard import (
    assert_process_consistency,
    assert_render_consistency,
    load_document_from_commits,
)
from bookextract.run_lock import acquire_run_lock
from bookextract.storage import RunStore


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _wire_schema_sha256() -> str:
    from bookextract.schema import load_wire_schema

    schema = load_wire_schema()
    return hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


_PYMUPDF_OPEN_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    RuntimeError,
    fitz.FileDataError,
    fitz.EmptyFileError,
)


def _validate_source_pdf(path: Path) -> tuple[str, int, int]:
    try:
        sha256, size_bytes = _hash_file(path)
        with fitz.open(path) as document:
            page_count = len(document)
    except _PYMUPDF_OPEN_ERRORS as exc:
        raise ProcessingError(
            code="invalid-source-pdf",
            message=f"cannot open source PDF: {path}",
        ) from exc

    if page_count < 1:
        raise ProcessingError(
            code="invalid-source-pdf",
            message="source PDF contains no pages",
        )

    return sha256, size_bytes, page_count


def _validate_init_inputs(
    *,
    pdf_path: Path,
    config_path: Path,
    model_path: Path,
    projector_path: Path | None,
) -> tuple[ProcessingConfig, str, int, int]:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"source PDF not found: {pdf_path}")
    config = load_processing_config(config_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"model file not found: {model_path}")
    if projector_path is not None and not projector_path.is_file():
        raise FileNotFoundError(f"projector file not found: {projector_path}")
    sha256, size_bytes, page_count = _validate_source_pdf(pdf_path)
    return config, sha256, size_bytes, page_count


def _compute_relocation(
    *,
    store: RunStore,
    location: InferenceLocation,
    model: Path | None,
    projector: Path | None,
    clear_projector: bool,
) -> InferenceLocation:
    resulting_model = model.resolve() if model is not None else location.model_file_path
    if clear_projector:
        resulting_projector: Path | None = None
    elif projector is not None:
        resulting_projector = projector.resolve()
    else:
        resulting_projector = location.projector_file_path

    if not resulting_model.is_file():
        raise ProcessingError(
            code="inference-relocation-rejected",
            message=f"model file not found: {resulting_model}",
        )
    if resulting_projector is not None and not resulting_projector.is_file():
        raise ProcessingError(
            code="inference-relocation-rejected",
            message=f"projector file not found: {resulting_projector}",
        )

    existing = store.load_inference_environment()
    if existing is None:
        return InferenceLocation(
            inference_location_format_version=1,
            model_file_path=resulting_model,
            projector_file_path=resulting_projector,
        )

    if (
        not existing.fingerprints_complete
        or existing.model_file is None
        or not existing.model_file.sha256
    ):
        raise ProcessingError(
            code="inference-relocation-rejected",
            message="inference environment fingerprints are incomplete",
        )

    stored_projector = existing.projector_file
    if stored_projector is not None and not stored_projector.sha256:
        raise ProcessingError(
            code="inference-relocation-rejected",
            message="incomplete projector fingerprint",
        )

    model_file = existing.model_file
    new_model_fp = fingerprint_file(resulting_model)
    if new_model_fp.sha256 != model_file.sha256:
        raise ProcessingError(
            code="inference-relocation-rejected",
            message="model fingerprint mismatch",
        )

    if stored_projector is not None and stored_projector.sha256:
        if resulting_projector is None:
            raise ProcessingError(
                code="inference-relocation-rejected",
                message="cannot clear projector after environment freeze",
            )
        new_projector_fp = fingerprint_file(resulting_projector)
        if new_projector_fp.sha256 != stored_projector.sha256:
            raise ProcessingError(
                code="inference-relocation-rejected",
                message="projector fingerprint mismatch",
            )
    elif resulting_projector is not None:
        raise ProcessingError(
            code="inference-relocation-rejected",
            message="cannot add projector after environment freeze",
        )

    return InferenceLocation(
        inference_location_format_version=1,
        model_file_path=resulting_model,
        projector_file_path=resulting_projector,
    )


def cmd_init(args: argparse.Namespace) -> int:
    pdf_path = Path(args.pdf).resolve()
    run_arg = Path(args.run)
    config_path = Path(args.config).resolve()
    model_path = Path(args.model).resolve()
    projector_path = Path(args.projector).resolve() if args.projector else None

    try:
        config, sha256, size_bytes, page_count = _validate_init_inputs(
            pdf_path=pdf_path,
            config_path=config_path,
            model_path=model_path,
            projector_path=projector_path,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (tomllib.TOMLDecodeError, ValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except BookExtractError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return exit_code_for_error(exc)

    with acquire_run_lock(run_arg):
        run_dir = run_arg.resolve()
        if (
            (run_dir / "run.json").exists()
            or (run_dir / "head.json").exists()
            or (run_dir / "commits").exists()
            or (run_dir / "inference-environment.json").exists()
        ):
            print("run-already-initialized", file=sys.stderr)
            return 13

        run_dir.mkdir(parents=True, exist_ok=True)
        store = RunStore(run_dir)
        store.ensure_layout()

        run_record = RunRecord(
            run_format_version=1,
            source={"sha256": sha256, "size_bytes": size_bytes, "page_count": page_count},
            extraction=config.extraction,
            fingerprint_policy=config.fingerprint,
            process_options=config.process,
            markdown=config.markdown,
            epub=config.epub,
            render_contract=RenderContract(
                render_contract_format_version=1,
                pymupdf_version=fitz.__version__,
            ),
            prompt_sha256=prompt_sha256(),
            wire_schema_sha256=_wire_schema_sha256(),
            created_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        )
        write_json_atomic(run_dir / "run.json", run_record.model_dump(mode="json"))

        store.write_source_location(
            SourceLocation(source_location_format_version=1, pdf_path=pdf_path)
        )
        store.write_inference_location(
            InferenceLocation(
                inference_location_format_version=1,
                model_file_path=model_path,
                projector_file_path=projector_path,
            )
        )
        store.write_head(0)
    print(f"initialized run at {run_dir}")
    return 0


def _process_kwargs(args: argparse.Namespace) -> dict[str, int | None]:
    kwargs: dict[str, int | None] = {}
    if args.through_page is not None:
        kwargs["through_page"] = args.through_page
    if args.max_pages is not None:
        kwargs["max_pages"] = args.max_pages
    return kwargs


def cmd_process(args: argparse.Namespace) -> int:
    run_arg = Path(args.run)

    try:
        with acquire_run_lock(run_arg):
            run_dir = run_arg.resolve()
            record = load_run_record(run_dir)
            config = processing_config_from_run_record(record)
            store = RunStore(run_dir)
            store.recover()
            recover_output_transaction(store, "markdown")
            recover_output_transaction(store, "epub")
            quarantine_invalid_output_destinations(store)
            assert_process_consistency(
                store,
                record,
                require_inference_location=args.interpreter == "vlm",
            )

            source_loc = store.load_source_location()
            pdf_source = PdfPageSource(
                source_loc.pdf_path,
                store._path("pages"),
                dpi=config.extraction.render_dpi,
            )
            try:
                if args.interpreter == "vlm":
                    from bookextract.context import build_page_context
                    from bookextract.inference.bootstrap import prepare_inference_environment
                    from bookextract.inference.llamacpp import LlamaCppVisionClient
                    from bookextract.interpretation.prompts import PagePromptBuilder
                    from bookextract.interpretation.vlm import (
                        LlamaCppStructuredClient,
                        VlmPageInterpreter,
                    )
                    from bookextract.models import RenderedPage
                    from bookextract.state import load_or_initialize_state

                    store.load_inference_location()
                    state, _ = load_or_initialize_state(store)
                    calibration_context = build_page_context(state)
                    calibration_prompt = PagePromptBuilder().build(calibration_context)

                    existing_env = store.load_inference_environment()
                    render_calibration_page: Callable[[], RenderedPage] | None = None
                    if existing_env is None:

                        def render_calibration_page() -> RenderedPage:
                            return pdf_source.render_page(
                                0, dpi=config.extraction.render_dpi
                            )

                    with LlamaCppVisionClient(config) as llama_client:
                        prepare_inference_environment(
                            store=store,
                            client=llama_client,
                            calibration_context=calibration_context,
                            calibration_prompt=calibration_prompt,
                            render_calibration_page=render_calibration_page,
                        )
                        interpreter = VlmPageInterpreter(
                            LlamaCppStructuredClient(llama_client),
                            model=config.extraction.model_alias,
                            prompt_version=config.extraction.prompt_version,
                        )
                        process_book(
                            source=pdf_source,
                            interpreter=interpreter,
                            store=store,
                            config=config,
                            **_process_kwargs(args),
                        )
                else:
                    from bookextract.artifacts import InterpretationResult
                    from bookextract.models import (
                        ExtractedMetadata,
                        InterpretationProvenance,
                        PageContext,
                        PageInput,
                        PageInterpretation,
                        PageType,
                    )

                    class NoopInterpreter:
                        def interpret(
                            self, *, page_input: PageInput, context: PageContext
                        ) -> InterpretationResult:
                            del page_input, context
                            return InterpretationResult(
                                interpretation=PageInterpretation(
                                    page_type=PageType.BLANK,
                                    metadata=ExtractedMetadata(),
                                ),
                                provenance=InterpretationProvenance(
                                    prompt_version=config.extraction.prompt_version,
                                    model=config.extraction.model_alias,
                                    backend="noop",
                                ),
                            )

                    process_book(
                        source=pdf_source,
                        interpreter=NoopInterpreter(),
                        store=store,
                        config=config,
                        **_process_kwargs(args),
                    )
            except BookExtractError as exc:
                print(f"{exc.code}: {exc}", file=sys.stderr)
                return exit_code_for_error(exc)
            finally:
                pdf_source.close()
    except BookExtractError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return exit_code_for_error(exc)
    return 0


def cmd_relocate_inference_files(args: argparse.Namespace) -> int:
    if not (args.model or args.projector or args.clear_projector):
        print(
            "at least one of --model, --projector, or --clear-projector is required",
            file=sys.stderr,
        )
        return 2
    if args.projector and args.clear_projector:
        print("--projector and --clear-projector are mutually exclusive", file=sys.stderr)
        return 2

    run_arg = Path(args.run)
    try:
        with acquire_run_lock(run_arg):
            run_dir = run_arg.resolve()
            store = RunStore(run_dir)
            location = store.load_inference_location()

            new_location = _compute_relocation(
                store=store,
                location=location,
                model=Path(args.model) if args.model else None,
                projector=Path(args.projector) if args.projector else None,
                clear_projector=args.clear_projector,
            )
            if (
                new_location.model_file_path == location.model_file_path
                and new_location.projector_file_path == location.projector_file_path
            ):
                print("inference-location.json unchanged")
                return 0
            store.write_inference_location(new_location)
    except BookExtractError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return exit_code_for_error(exc)

    print("updated inference-location.json")
    return 0


def cmd_render_markdown(args: argparse.Namespace) -> int:
    run_arg = Path(args.run)
    try:
        with acquire_run_lock(run_arg):
            run_dir = run_arg.resolve()
            record = load_run_record(run_dir)
            config = processing_config_from_run_record(record)
            store = RunStore(run_dir)
            store.recover()
            recover_output_transaction(store, "markdown")
            assert_render_consistency(store, record, "markdown")
            document = load_document_from_commits(store)
            head = store.read_head().committed_page_count

            pub_doc, source_map = build_publication_document(
                document,
                markdown_config=config.markdown,
                epub_config=config.epub,
            )
            renderer = MarkdownRenderer(config.markdown)
            markdown = renderer.render_publication(pub_doc).encode("utf-8")
            assets = collect_commit_assets(store)
            manifest = build_output_manifest(
                command="markdown",
                committed_page_count=head,
                publication=pub_doc,
                source_map=source_map,
                files=[],
            )

            def builder(candidate_path: Path, _work_path: Path | None) -> None:
                write_output_bundle(
                    candidate_path,
                    command="markdown",
                    primary_name="book.md",
                    primary_bytes=markdown,
                    manifest=manifest,
                    assets=assets,
                )

            destination = publish_output(store, "markdown", builder)
    except BookExtractError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return exit_code_for_error(exc)

    print(f"wrote {destination / 'book.md'}")
    return 0


def cmd_render_epub(args: argparse.Namespace) -> int:
    run_arg = Path(args.run)
    try:
        with acquire_run_lock(run_arg):
            run_dir = run_arg.resolve()
            record = load_run_record(run_dir)
            config = processing_config_from_run_record(record)
            store = RunStore(run_dir)
            store.recover()
            recover_output_transaction(store, "epub")
            assert_render_consistency(store, record, "epub")
            document = load_document_from_commits(store)
            head = store.read_head().committed_page_count

            pub_doc, source_map = build_publication_document(
                document,
                markdown_config=config.markdown,
                epub_config=config.epub,
            )
            renderer = MarkdownRenderer(config.markdown)
            markdown = renderer.render_publication(pub_doc)
            assets = collect_commit_assets(store)
            manifest = build_output_manifest(
                command="epub",
                committed_page_count=head,
                publication=pub_doc,
                source_map=source_map,
                files=[],
            )
            epub_renderer = EpubRenderer(config.epub)

            def builder(candidate_path: Path, work_path: Path | None) -> None:
                if work_path is None:
                    raise ProcessingError(
                        code="invalid-run-layout",
                        message="epub render requires a work directory",
                    )
                markdown_path = work_path / "book.md"
                markdown_path.write_text(markdown, encoding="utf-8")
                assets_dir = work_path / "assets"
                if assets_dir.exists():
                    shutil.rmtree(assets_dir)
                for rel_path, content in assets.items():
                    dest = assets_dir / Path(rel_path)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(content)
                epub_path = candidate_path / "book.epub"
                epub_renderer.render(
                    markdown_path=markdown_path,
                    output_path=epub_path,
                    metadata=pub_doc.metadata,
                    build_directory=work_path,
                )
                epub_renderer.run_epubcheck(epub_path)
                write_output_bundle(
                    candidate_path,
                    command="epub",
                    primary_name="book.epub",
                    primary_bytes=epub_path.read_bytes(),
                    manifest=manifest,
                    assets=assets,
                )

            destination = publish_output(
                store,
                "epub",
                builder,
                use_work_directory=True,
            )
    except BookExtractError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return exit_code_for_error(exc)

    print(f"wrote {destination / 'book.epub'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bookextract")
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init", help="initialize a new run directory")
    init_p.add_argument("pdf", type=Path)
    init_p.add_argument("--run", type=Path, required=True)
    init_p.add_argument("--config", type=Path, required=True)
    init_p.add_argument("--model", type=Path, required=True)
    init_p.add_argument("--projector", type=Path, default=None)

    process_p = sub.add_parser("process", help="process book pages")
    process_p.add_argument("--run", type=Path, required=True)
    process_p.add_argument("--interpreter", choices=["noop", "vlm"], default="vlm")
    page_group = process_p.add_mutually_exclusive_group()
    page_group.add_argument("--through-page", type=int, default=None)
    page_group.add_argument("--max-pages", type=int, default=None)

    relocate_p = sub.add_parser(
        "relocate-inference-files",
        help="update inference model/projector paths",
    )
    relocate_p.add_argument("--run", type=Path, required=True)
    relocate_p.add_argument("--model", type=Path, default=None)
    relocate_p.add_argument("--projector", type=Path, default=None)
    relocate_p.add_argument("--clear-projector", action="store_true")

    md_p = sub.add_parser("render-markdown", help="render committed pages to Markdown")
    md_p.add_argument("--run", type=Path, required=True)

    epub_p = sub.add_parser("render-epub", help="render committed pages to EPUB")
    epub_p.add_argument("--run", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "init": cmd_init,
        "process": cmd_process,
        "relocate-inference-files": cmd_relocate_inference_files,
        "render-markdown": cmd_render_markdown,
        "render-epub": cmd_render_epub,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
