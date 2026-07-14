"""Figure asset cropping tests."""

from __future__ import annotations

from pathlib import Path

import fitz

from bookextract.assets import crop_figure_asset
from bookextract.models import BoundingBox, RectPoints, RenderedPage


def test_crop_figure_asset_from_rendered_page(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.draw_rect(fitz.Rect(40, 40, 160, 160), color=(1, 0, 0), fill=(1, 0, 0))
    pix = page.get_pixmap()
    pix.save(str(image_path))
    doc.close()

    rendered = RenderedPage(
        image_path=image_path,
        width_px=pix.width,
        height_px=pix.height,
        page_rect=RectPoints(x0=0, y0=0, x1=200, y1=200),
        rotation_degrees=0,
        image_sha256="0" * 64,
        image_size_bytes=image_path.stat().st_size,
    )
    png_bytes, sha = crop_figure_asset(
        rendered,
        BoundingBox(x0=0.2, y0=0.2, x1=0.8, y1=0.8),
    )
    assert len(png_bytes) > 0
    assert len(sha) == 64
