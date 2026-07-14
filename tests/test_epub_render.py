"""EPUB rendering and Pandoc defaults tests."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from bookextract.errors import ProcessingError
from bookextract.rendering.epub import EpubRenderer, _validate_base_defaults


def _fake_pandoc_run(
    command: list[str],
    *,
    cwd: str | Path,
    asset_name: str,
    asset_bytes: bytes,
    **kwargs: object,
) -> subprocess.CompletedProcess[str]:
    del kwargs
    cwd_path = Path(cwd)
    if (cwd_path / "assets" / asset_name).read_bytes() != asset_bytes:
        raise AssertionError("asset missing or wrong at Pandoc invocation")
    (cwd_path / "book.epub").write_bytes(b"epub")
    return subprocess.CompletedProcess(command, 0, "", "")


def test_validate_base_defaults_accepts_frozen_mapping() -> None:
    result = _validate_base_defaults(
        {
            "from": "markdown+footnotes",
            "to": "epub3",
            "split-level": 1,
        }
    )
    assert result["from"] == "markdown+footnotes"


@pytest.mark.parametrize(
    "payload",
    [
        {"from": "commonmark", "to": "epub3", "split-level": 1},
        {"from": "markdown+footnotes", "to": "epub2", "split-level": 1},
        {"from": "markdown+footnotes", "to": "epub3", "split-level": 4},
        {"from": "markdown+footnotes", "to": "epub3", "split-level": True},
        {"from": "markdown+footnotes", "to": "epub3", "split-level": 1.0},
        ["markdown+footnotes", "epub3", 1],
    ],
    ids=["wrong-from", "wrong-to", "wrong-split", "split-bool", "split-float", "array-top"],
)
def test_validate_base_defaults_rejects_invalid(payload: object) -> None:
    with pytest.raises(ProcessingError, match="unsupported-pandoc-defaults"):
        _validate_base_defaults(payload)


def test_load_base_defaults_rejects_malformed_json(tmp_path: Path) -> None:
    defaults_path = tmp_path / "defaults.json"
    defaults_path.write_text("{not json", encoding="utf-8")
    renderer = EpubRenderer(pandoc_defaults_path=defaults_path)
    with pytest.raises(ProcessingError, match="cannot load frozen Pandoc defaults"):
        renderer._load_base_defaults()


def test_render_preserves_colocated_assets_at_pandoc_invocation(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    asset_name = "a" * 64 + ".png"
    asset_bytes = b"figure-png"
    assets_dir = work_dir / "assets"
    assets_dir.mkdir()
    (assets_dir / asset_name).write_bytes(asset_bytes)
    markdown_path = work_dir / "book.md"
    markdown_path.write_text("# Title\n", encoding="utf-8")
    output_path = tmp_path / "out" / "book.epub"

    renderer = EpubRenderer()
    with patch(
        "bookextract.rendering.epub.subprocess.run",
        side_effect=lambda command, *, cwd, **kwargs: _fake_pandoc_run(
            command,
            cwd=cwd,
            asset_name=asset_name,
            asset_bytes=asset_bytes,
            **kwargs,
        ),
    ):
        renderer.render(
            markdown_path=markdown_path,
            output_path=output_path,
            build_directory=work_dir,
        )

    assert output_path.read_bytes() == b"epub"
    assert (assets_dir / asset_name).read_bytes() == asset_bytes


def test_render_copies_assets_from_separate_source_directory(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    asset_name = "b" * 64 + ".png"
    asset_bytes = b"remote-figure"
    assets_src = source_dir / "assets"
    assets_src.mkdir()
    (assets_src / asset_name).write_bytes(asset_bytes)
    markdown_path = source_dir / "book.md"
    markdown_path.write_text("# Title\n", encoding="utf-8")

    build_dir = tmp_path / "build"
    build_dir.mkdir()
    output_path = tmp_path / "out" / "book.epub"

    renderer = EpubRenderer()
    with patch(
        "bookextract.rendering.epub.subprocess.run",
        side_effect=lambda command, *, cwd, **kwargs: _fake_pandoc_run(
            command,
            cwd=cwd,
            asset_name=asset_name,
            asset_bytes=asset_bytes,
            **kwargs,
        ),
    ):
        renderer.render(
            markdown_path=markdown_path,
            output_path=output_path,
            build_directory=build_dir,
        )

    assert (build_dir / "assets" / asset_name).read_bytes() == asset_bytes


def test_stage_publication_assets_uses_canonical_paths(tmp_path: Path) -> None:
    from bookextract.cli import _stage_publication_assets
    from bookextract.output_paths import figure_asset_path

    work_path = tmp_path / "epub.work.test"
    work_path.mkdir()
    sha256 = "c" * 64
    png_bytes = b"figure-png"
    assets = {figure_asset_path(sha256): png_bytes}

    _stage_publication_assets(work_path, assets)

    staged = work_path / figure_asset_path(sha256)
    assert staged.read_bytes() == png_bytes
    assert not (work_path / "assets" / "assets").exists()
