"""Image-plane spatial relationships from bounding boxes (report section 5.5).

These describe positions in the 2D picture. They are never physical distances.
"""

from __future__ import annotations

from itertools import combinations

from .config import DEFAULT_CONFIG, Config
from .evidence import BBox, Detection, SpatialRelation


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def horizontal_region(bbox: BBox, image_width: int) -> str:
    centre = (bbox[0] + bbox[2]) / 2 / image_width
    if centre < 1 / 3:
        return "left"
    if centre > 2 / 3:
        return "right"
    return "center"


def relation_between(a: BBox, b: BBox, overlap_iou: float) -> str:
    """Relation of box a to box b, using whichever axis separates them more."""
    if iou(a, b) >= overlap_iou:
        return "overlaps"
    dx = (a[0] + a[2]) / 2 - (b[0] + b[2]) / 2
    dy = (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2  # image y grows downward
    if abs(dx) >= abs(dy):
        return "left of" if dx < 0 else "right of"
    return "above" if dy < 0 else "below"


def analyze_spatial(
    detections: list[Detection], config: Config = DEFAULT_CONFIG
) -> list[SpatialRelation]:
    """Pairwise relations among the most confident objects, to keep the output readable."""
    top = sorted(detections, key=lambda d: d.confidence, reverse=True)[: config.max_spatial_objects]
    relations = []
    for a, b in combinations(top, 2):
        relation = relation_between(a.bbox, b.bbox, config.overlap_iou)
        relations.append(
            SpatialRelation(
                subject_id=a.id,
                relation=relation,
                object_id=b.id,
                description=f"{a.label} ({a.id}) {relation} {b.label} ({b.id}) in the image",
            )
        )
    return relations
