"""Figure asset extraction from rendered pages."""

from __future__ import annotations

import hashlib
from typing import cast

import fitz

from bookextract.models import BoundingBox, RenderedPage
from bookextract.pdf import bbox1000_to_pixel_rect, normalized_bbox_to_pixel_rect
from bookextract.wire import BBox1000


def _clamp_irect(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    width: int,
    height: int,
) -> fitz.Rect:
    left = max(0, min(x0, x1))
    top = max(0, min(y0, y1))
    right = min(width, max(x0, x1))
    bottom = min(height, max(y0, y1))
    if right <= left or bottom <= top:
        raise ValueError("degenerate crop rectangle")
    return fitz.Rect(left, top, right, bottom)


def _crop_pixmap(rendered: RenderedPage, x0: int, y0: int, x1: int, y1: int) -> bytes:
    pixmap = fitz.Pixmap(str(rendered.image_path))
    try:
        clip = _clamp_irect(
            x0,
            y0,
            x1,
            y1,
            width=pixmap.width,
            height=pixmap.height,
        )
        cropped = fitz.Pixmap(
            fitz.csRGB,
            fitz.IRect(0, 0, int(clip.width), int(clip.height)),
            False,
        )
        cropped.copy(pixmap, clip)
        return cast(bytes, cropped.tobytes("png"))
    finally:
        pixmap = None


def crop_figure_from_bbox1000(rendered: RenderedPage, bbox: BBox1000) -> bytes:
    """Crop a figure region from a rendered page using wire bbox coordinates."""
    pixel = bbox1000_to_pixel_rect(bbox, rendered.width_px, rendered.height_px)
    return _crop_pixmap(
        rendered,
        int(pixel.x0),
        int(pixel.y0),
        int(pixel.x1),
        int(pixel.y1),
    )


def crop_figure_from_bbox(rendered: RenderedPage, bbox: BoundingBox) -> bytes:
    """Crop a figure region from a rendered page using normalized domain bbox."""
    pixel = normalized_bbox_to_pixel_rect(bbox, rendered.width_px, rendered.height_px)
    return _crop_pixmap(
        rendered,
        int(pixel.x0),
        int(pixel.y0),
        int(pixel.x1),
        int(pixel.y1),
    )


def crop_figure_asset(
    rendered: RenderedPage,
    bbox: BoundingBox | BBox1000,
) -> tuple[bytes, str]:
    """Crop figure bytes and return ``(png_bytes, sha256_hex)``."""
    if isinstance(bbox, BBox1000):
        png_bytes = crop_figure_from_bbox1000(rendered, bbox)
    else:
        png_bytes = crop_figure_from_bbox(rendered, bbox)
    return png_bytes, hashlib.sha256(png_bytes).hexdigest()
