"""PDF coordinate transform tests."""

from __future__ import annotations

from bookextract.models import BoundingBox
from bookextract.pdf import bbox1000_to_pixel_rect, normalized_bbox_to_pixel_rect
from bookextract.wire import BBox1000


def test_bbox1000_to_pixel_rect() -> None:
    rect = bbox1000_to_pixel_rect(BBox1000(left=0, top=0, right=1000, bottom=1000), 2000, 1000)
    assert rect.x0 == 0.0
    assert rect.y0 == 0.0
    assert rect.x1 == 2000.0
    assert rect.y1 == 1000.0


def test_normalized_bbox_to_pixel_rect() -> None:
    rect = normalized_bbox_to_pixel_rect(
        BoundingBox(x0=0.25, y0=0.5, x1=0.75, y1=1.0),
        width_px=100,
        height_px=200,
    )
    assert rect.x0 == 25.0
    assert rect.y0 == 100.0
    assert rect.x1 == 75.0
    assert rect.y1 == 200.0
