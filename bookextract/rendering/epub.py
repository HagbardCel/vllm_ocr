"""EPUB rendering via Pandoc."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from bookextract.config import EpubRenderConfig
from bookextract.errors import ExternalToolError, ProcessingError
from bookextract.models import PublicationDocument, PublicationMetadata

REQUIRED_BASE_DEFAULTS: dict[str, object] = {
    "from": "markdown+footnotes",
    "to": "epub3",
    "split-level": 1,
}


def _validate_base_defaults(base: object) -> dict[str, object]:
    if not isinstance(base, dict):
        raise ProcessingError(code="unsupported-pandoc-defaults")

    if set(base) != set(REQUIRED_BASE_DEFAULTS):
        raise ProcessingError(code="unsupported-pandoc-defaults")

    for key, expected in REQUIRED_BASE_DEFAULTS.items():
        actual = base[key]
        if type(actual) is not type(expected) or actual != expected:
            raise ProcessingError(code="unsupported-pandoc-defaults")

    return base


def epubcheck_argv(
    config: EpubRenderConfig,
    epub_path: Path,
    report_path: Path,
) -> list[str]:
    if config.epubcheck_jar_path is not None:
        return [
            "java",
            "-jar",
            str(config.epubcheck_jar_path.resolve()),
            "--json",
            str(report_path.resolve()),
            str(epub_path.resolve()),
        ]
    return [
        config.epubcheck_executable,
        "--json",
        str(report_path.resolve()),
        str(epub_path.resolve()),
    ]


class EpubRenderer:
    def __init__(
        self,
        config: EpubRenderConfig | None = None,
        *,
        pandoc_defaults_path: Path | None = None,
    ) -> None:
        self._config = config or EpubRenderConfig()
        self._defaults_path = pandoc_defaults_path

    def _load_base_defaults(self) -> dict[str, object]:
        try:
            if self._defaults_path is None:
                from importlib import resources

                text = resources.files("bookextract.resources").joinpath(
                    "pandoc-epub-v1.json"
                ).read_text(encoding="utf-8")
            else:
                text = self._defaults_path.read_text(encoding="utf-8")
            raw = json.loads(text)
        except (OSError, ValueError, TypeError) as exc:
            raise ProcessingError(
                code="unsupported-pandoc-defaults",
                message="cannot load frozen Pandoc defaults",
            ) from exc
        return _validate_base_defaults(raw)

    def run_epubcheck(self, epub_path: Path, *, report_path: Path) -> None:
        command = epubcheck_argv(self._config, epub_path, report_path)
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise ExternalToolError(
                code="epubcheck-failed",
                message=exc.stderr or exc.stdout or str(exc),
            ) from exc
        except OSError as exc:
            raise ExternalToolError(
                code="epubcheck-failed",
                message=f"cannot execute EPUBCheck: {exc}",
            ) from exc

    def render(
        self,
        *,
        markdown_path: Path,
        output_path: Path,
        metadata: PublicationMetadata | None = None,
        build_directory: Path | None = None,
    ) -> None:
        base = self._load_base_defaults()
        build_dir = build_directory or output_path.parent / ".output-build"
        build_dir.mkdir(parents=True, exist_ok=True)

        meta: dict[str, str] = {}
        if metadata:
            if metadata.title:
                meta["title"] = metadata.title
            if metadata.subtitle:
                meta["subtitle"] = metadata.subtitle
            if metadata.authors:
                meta["author"] = "; ".join(metadata.authors)
            if metadata.language:
                meta["lang"] = metadata.language

        pandoc_build = {
            **base,
            "toc": self._config.include_toc,
            "metadata": meta,
        }
        defaults_file = build_dir / "pandoc-build.json"
        defaults_file.write_text(
            json.dumps(pandoc_build, indent=2) + "\n", encoding="utf-8"
        )

        markdown_in_build = build_dir / "book.md"
        markdown_in_build.write_bytes(markdown_path.read_bytes())

        assets_src = markdown_path.parent / "assets"
        assets_dest = build_dir / "assets"
        if assets_src.is_dir() and assets_src.resolve() != assets_dest.resolve():
            if assets_dest.exists():
                shutil.rmtree(assets_dest)
            shutil.copytree(assets_src, assets_dest)

        command = [
            self._config.pandoc_executable,
            "--defaults",
            str(defaults_file.resolve()),
            "book.md",
            "--resource-path",
            ".",
            "--output",
            "book.epub",
        ]
        try:
            subprocess.run(
                command,
                cwd=build_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise ExternalToolError(
                code="pandoc-failed",
                message=exc.stderr or str(exc),
            ) from exc
        except OSError as exc:
            raise ExternalToolError(
                code="pandoc-failed",
                message=f"cannot execute Pandoc: {exc}",
            ) from exc

        produced = build_dir / "book.epub"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(produced.read_bytes())

    def render_publication(
        self,
        *,
        publication: PublicationDocument,
        markdown: str,
        output_path: Path,
        build_directory: Path,
    ) -> None:
        build_directory.mkdir(parents=True, exist_ok=True)
        markdown_path = build_directory / "book.md"
        markdown_path.write_text(markdown, encoding="utf-8")
        self.render(
            markdown_path=markdown_path,
            output_path=output_path,
            metadata=publication.metadata,
            build_directory=build_directory,
        )
