"""PDF page rendering and coordinate transforms."""

from __future__ import annotations

import hashlib
from pathlib import Path

import fitz

from bookextract.errors import ProcessingError
from bookextract.models import (
    MAX_PAGE_IMAGE_BYTES,
    BoundingBox,
    RectPoints,
    RenderedPage,
)
from bookextract.wire import BBox1000

__all__ = [
    "MAX_PAGE_IMAGE_BYTES",
    "PdfPageSource",
    "RenderedPage",
    "bbox1000_to_pdf_rect",
    "bbox1000_to_pixel_rect",
    "normalized_bbox_to_pixel_rect",
    "pixel_rect_to_pdf_rect",
]


def _clamp_rect(x0: float, y0: float, x1: float, y1: float) -> RectPoints:
    left = min(x0, x1)
    right = max(x0, x1)
    top = min(y0, y1)
    bottom = max(y0, y1)
    return RectPoints(x0=left, y0=top, x1=right, y1=bottom)


def bbox1000_to_pixel_rect(bbox: BBox1000, width_px: int, height_px: int) -> RectPoints:
    """Map VLM bbox coordinates on [0, 1000] to rendered-image pixel space."""
    return _clamp_rect(
        bbox.left * width_px / 1000.0,
        bbox.top * height_px / 1000.0,
        bbox.right * width_px / 1000.0,
        bbox.bottom * height_px / 1000.0,
    )


def normalized_bbox_to_pixel_rect(
    bbox: BoundingBox,
    width_px: int,
    height_px: int,
) -> RectPoints:
    """Map normalized [0, 1] domain bbox to rendered-image pixel space."""
    return _clamp_rect(
        bbox.x0 * width_px,
        bbox.y0 * height_px,
        bbox.x1 * width_px,
        bbox.y1 * height_px,
    )


def _display_point_to_pdf_point(
    x_px: float,
    y_px: float,
    *,
    page: fitz.Page,
    rendered: RenderedPage,
) -> tuple[float, float]:
    rect = page.rect
    width_px = rendered.width_px
    height_px = rendered.height_px
    rotation = rendered.rotation_degrees % 360

    if rotation == 0:
        scale_x = rect.width / width_px
        scale_y = rect.height / height_px
        return rect.x0 + x_px * scale_x, rect.y0 + y_px * scale_y
    if rotation == 90:
        return (
            rect.x0 + (1.0 - y_px / height_px) * rect.width,
            rect.y0 + (x_px / width_px) * rect.height,
        )
    if rotation == 180:
        return (
            rect.x0 + (1.0 - x_px / width_px) * rect.width,
            rect.y0 + (1.0 - y_px / height_px) * rect.height,
        )
    if rotation == 270:
        return (
            rect.x0 + (y_px / height_px) * rect.width,
            rect.y0 + (1.0 - x_px / width_px) * rect.height,
        )
    raise ValueError(f"unsupported page rotation: {rotation}")


def pixel_rect_to_pdf_rect(
    pixel: RectPoints,
    page: fitz.Page,
    rendered: RenderedPage,
) -> RectPoints:
    """Map a pixel rectangle on the rendered image back to PDF page points."""
    corners = [
        _display_point_to_pdf_point(pixel.x0, pixel.y0, page=page, rendered=rendered),
        _display_point_to_pdf_point(pixel.x1, pixel.y0, page=page, rendered=rendered),
        _display_point_to_pdf_point(pixel.x0, pixel.y1, page=page, rendered=rendered),
        _display_point_to_pdf_point(pixel.x1, pixel.y1, page=page, rendered=rendered),
    ]
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return _clamp_rect(min(xs), min(ys), max(xs), max(ys))


def bbox1000_to_pdf_rect(
    bbox: BBox1000,
    page: fitz.Page,
    rendered: RenderedPage,
) -> RectPoints:
    """Full coordinate pipeline from VLM bbox to PDF cropbox space."""
    pixel = bbox1000_to_pixel_rect(bbox, rendered.width_px, rendered.height_px)
    return pixel_rect_to_pdf_rect(pixel, page, rendered)


class PdfPageSource:
    def __init__(
        self,
        pdf_path: Path,
        pages_dir: Path,
        *,
        dpi: int = 240,
        render_annotations: bool = False,
    ) -> None:
        self._pdf_path = pdf_path.resolve()
        self._pages_dir = pages_dir
        self._dpi = dpi
        self._render_annotations = render_annotations
        self._doc = fitz.open(self._pdf_path)

    @property
    def pdf_path(self) -> Path:
        return self._pdf_path

    @property
    def page_count(self) -> int:
        return len(self._doc)

    def close(self) -> None:
        self._doc.close()

    def __enter__(self) -> PdfPageSource:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def render_page(self, page_index: int, *, dpi: int | None = None) -> RenderedPage:
        try:
            return self._render_page_impl(page_index, dpi=dpi)
        except ProcessingError:
            raise
        except Exception as exc:
            raise ProcessingError(
                code="page-render-failed",
                message=f"failed to render page {page_index}",
            ) from exc

    def _render_page_impl(self, page_index: int, *, dpi: int | None = None) -> RenderedPage:
        render_dpi = dpi if dpi is not None else self._dpi
        self._pages_dir.mkdir(parents=True, exist_ok=True)
        image_path = self._pages_dir / f"page-{page_index + 1:04d}.png"

        page = self._doc.load_page(page_index)
        rect = page.rect
        width_px: int
        height_px: int

        if not image_path.is_file():
            matrix = fitz.Matrix(render_dpi / 72, render_dpi / 72)
            pix = page.get_pixmap(
                matrix=matrix,
                alpha=False,
                annots=self._render_annotations,
            )
            png_bytes = pix.tobytes("png")
            width_px = pix.width
            height_px = pix.height
            if len(png_bytes) > MAX_PAGE_IMAGE_BYTES:
                raise ProcessingError(
                    code="page-image-too-large",
                    message=f"rendered page exceeds {MAX_PAGE_IMAGE_BYTES} bytes",
                )
            image_path.write_bytes(png_bytes)
        else:
            png_bytes = image_path.read_bytes()
            with fitz.open(image_path) as image_doc:
                image_page = image_doc[0]
                width_px = int(image_page.rect.width)
                height_px = int(image_page.rect.height)

        return RenderedPage(
            image_path=image_path,
            width_px=width_px,
            height_px=height_px,
            page_rect=RectPoints(x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1),
            rotation_degrees=page.rotation,
            image_sha256=hashlib.sha256(png_bytes).hexdigest(),
            image_size_bytes=len(png_bytes),
        )
