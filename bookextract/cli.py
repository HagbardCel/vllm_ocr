"""bookextract command-line interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import fitz

from bookextract.canonical import (
    normalize_publication_document_nfc,
    publication_fingerprint_hex,
    publication_identifier,
)
from bookextract.config import (
    InferenceLocation,
    ProcessingConfig,
    RenderContract,
    RunRecord,
    SourceLocation,
    load_processing_config,
    load_run_record,
    write_json_atomic,
)
from bookextract.errors import BookExtractError, exit_code_for_error
from bookextract.interpretation.prompts import prompt_sha256
from bookextract.models import BookDocument, OutputManifest, PageAssessment
from bookextract.pdf import PdfPageSource
from bookextract.pipeline import process_book
from bookextract.rendering.epub import EpubRenderer
from bookextract.rendering.markdown import MarkdownRenderer
from bookextract.rendering.publication import build_publication_document
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


def cmd_init(args: argparse.Namespace) -> int:
    pdf_path = Path(args.pdf).resolve()
    run_dir = Path(args.run).resolve()
    config_path = Path(args.config).resolve()

    if not pdf_path.is_file():
        print(f"source PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    config = load_processing_config(config_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    with acquire_run_lock(run_dir):
        store = RunStore(run_dir)
        if (
            (run_dir / "run.json").exists()
            or (run_dir / "head.json").exists()
            or (run_dir / "commits").exists()
            or (run_dir / "inference-environment.json").exists()
        ):
            print("run-already-initialized", file=sys.stderr)
            return 13

        store.ensure_layout()

        sha256, size_bytes = _hash_file(pdf_path)
        with fitz.open(pdf_path) as doc:
            page_count = len(doc)

        run_record = RunRecord(
            source={"sha256": sha256, "size_bytes": size_bytes, "page_count": page_count},
            extraction=config.extraction,
            fingerprint_policy=config.fingerprint,
            render_contract=RenderContract(pymupdf_version=fitz.__version__),
            prompt_sha256=prompt_sha256(),
            wire_schema_sha256=_wire_schema_sha256(),
            created_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        )
        write_json_atomic(run_dir / "run.json", run_record.model_dump(mode="json"))

        store.write_source_location(
            SourceLocation(pdf_path=pdf_path),
        )
        if args.model:
            store.write_inference_location(
                InferenceLocation(
                    model_file_path=Path(args.model).resolve(),
                    projector_file_path=Path(args.projector).resolve() if args.projector else None,
                )
            )
        elif args.projector:
            print("--projector requires --model", file=sys.stderr)
            return 2

        store.write_head(0)
    print(f"initialized run at {run_dir}")
    return 0


def _load_document_from_commits(store: RunStore) -> BookDocument:
    head = store.read_head()
    document = BookDocument()
    for page_number in range(1, head.committed_page_count + 1):
        raw = store.read_commit_file(page_number, "page-assessment.json")
        document.pages.append(PageAssessment.model_validate_json(raw.decode("utf-8")))
    return document


def _resolve_config(args: argparse.Namespace, run_dir: Path) -> ProcessingConfig:
    if args.config:
        return load_processing_config(Path(args.config))
    run_record = load_run_record(run_dir)
    return ProcessingConfig(extraction=run_record.extraction)


def _process_kwargs(args: argparse.Namespace) -> dict[str, int | None]:
    kwargs: dict[str, int | None] = {}
    if args.through_page is not None:
        kwargs["through_page"] = args.through_page
    if args.max_pages is not None:
        kwargs["max_pages"] = args.max_pages
    return kwargs


def cmd_process(args: argparse.Namespace) -> int:
    run_dir = Path(args.run).resolve()
    config = _resolve_config(args, run_dir)

    try:
        with acquire_run_lock(run_dir):
            store = RunStore(run_dir)
            store.recover()

            if args.model or args.projector:
                location = store.load_inference_location()
                if args.model:
                    location = location.model_copy(
                        update={"model_file_path": Path(args.model).resolve()}
                    )
                if args.projector:
                    location = location.model_copy(
                        update={"projector_file_path": Path(args.projector).resolve()}
                    )
                store.write_inference_location(location)

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
                    from bookextract.state import load_or_initialize_state

                    store.load_inference_location()
                    calibration_page = pdf_source.render_page(
                        0, dpi=config.extraction.render_dpi
                    )
                    state, _ = load_or_initialize_state(store)
                    calibration_prompt = PagePromptBuilder().build(
                        build_page_context(state)
                    )

                    with LlamaCppVisionClient(config) as llama_client:
                        prepare_inference_environment(
                            store=store,
                            client=llama_client,
                            calibration_image=calibration_page.image_path,
                            calibration_prompt=calibration_prompt,
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
    run_dir = Path(args.run).resolve()
    store = RunStore(run_dir)
    location = store.load_inference_location()
    if args.model:
        location = location.model_copy(update={"model_file_path": Path(args.model).resolve()})
    if args.projector:
        location = location.model_copy(
            update={"projector_file_path": Path(args.projector).resolve()}
        )
    store.write_inference_location(location)
    print("updated inference-location.json")
    return 0


def cmd_render_markdown(args: argparse.Namespace) -> int:
    run_dir = Path(args.run).resolve()
    store = RunStore(run_dir)
    config = _resolve_config(args, run_dir)
    document = _load_document_from_commits(store)

    renderer = MarkdownRenderer(config.markdown)
    markdown = renderer.render_book(document, epub_include_toc=config.epub.include_toc)

    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "book.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    pub_doc, source_map = build_publication_document(
        document,
        markdown_config=config.markdown,
        epub_config=config.epub,
    )
    normalized = normalize_publication_document_nfc(pub_doc)
    manifest = OutputManifest(
        publication_identifier=publication_identifier(normalized),
        publication_fingerprint=publication_fingerprint_hex(normalized),
        source_map=sorted(source_map, key=lambda e: e.publication_block_index),
    )
    write_json_atomic(
        output_dir / "manifest.json",
        manifest.model_dump(mode="json"),
    )
    print(f"wrote {markdown_path}")
    return 0


def cmd_render_epub(args: argparse.Namespace) -> int:
    run_dir = Path(args.run).resolve()
    store = RunStore(run_dir)
    config = _resolve_config(args, run_dir)
    document = _load_document_from_commits(store)

    renderer = MarkdownRenderer(config.markdown)
    markdown = renderer.render_book(document, epub_include_toc=config.epub.include_toc)
    pub_doc, _ = build_publication_document(
        document,
        markdown_config=config.markdown,
        epub_config=config.epub,
    )

    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "book.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    epub_renderer = EpubRenderer(config.epub)
    try:
        epub_renderer.render_publication(
            publication=pub_doc,
            markdown=markdown,
            output_path=output_dir / "book.epub",
            build_directory=run_dir / ".output-build",
        )
    except BookExtractError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return exit_code_for_error(exc)

    print(f"wrote {output_dir / 'book.epub'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bookextract")
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init", help="initialize a new run directory")
    init_p.add_argument("pdf", type=Path)
    init_p.add_argument("--run", type=Path, required=True)
    init_p.add_argument("--config", type=Path, required=True)
    init_p.add_argument("--model", type=Path, default=None)
    init_p.add_argument("--projector", type=Path, default=None)

    process_p = sub.add_parser("process", help="process book pages")
    process_p.add_argument("--run", type=Path, required=True)
    process_p.add_argument("--config", type=Path, default=None)
    process_p.add_argument("--model", type=Path, default=None)
    process_p.add_argument("--projector", type=Path, default=None)
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

    md_p = sub.add_parser("render-markdown", help="render committed pages to Markdown")
    md_p.add_argument("--run", type=Path, required=True)
    md_p.add_argument("--config", type=Path, default=None)

    epub_p = sub.add_parser("render-epub", help="render committed pages to EPUB")
    epub_p.add_argument("--run", type=Path, required=True)
    epub_p.add_argument("--config", type=Path, default=None)

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
