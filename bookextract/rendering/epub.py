"""EPUB rendering via Pandoc."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import cast

from bookextract.config import EpubRenderConfig
from bookextract.errors import ExternalToolError, ProcessingError
from bookextract.models import PublicationDocument, PublicationMetadata

REQUIRED_BASE_DEFAULT_KEYS = frozenset({"from", "to", "split-level"})


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
        if self._defaults_path is None:
            package_root = Path(__file__).resolve().parents[2]
            path = package_root / "resources" / "pandoc-epub-v1.json"
        else:
            path = self._defaults_path
        base = json.loads(path.read_text(encoding="utf-8"))
        if set(base.keys()) != REQUIRED_BASE_DEFAULT_KEYS:
            raise ProcessingError(code="unsupported-pandoc-defaults")
        return cast(dict[str, object], base)

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
        if assets_src.is_dir():
            if assets_dest.exists():
                import shutil

                shutil.rmtree(assets_dest)
            import shutil

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
