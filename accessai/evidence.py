"""Data structures for the evidence record: the core of the audit trail."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

BBox = list[float]  # [x1, y1, x2, y2] in pixels of the preprocessed image

ClaimStatus = Literal["supported", "weak", "not_confirmed", "unverifiable", "inference"]


@dataclass
class Detection:
    id: str
    label: str
    confidence: float
    bbox: BBox
    region: str  # horizontal position in the image: left, center or right


@dataclass
class TextItem:
    id: str
    text: str
    confidence: float
    bbox: BBox


@dataclass
class SpatialRelation:
    subject_id: str
    relation: str  # left of, right of, above, below or overlaps
    object_id: str
    description: str


@dataclass
class Caption:
    text: str
    confidence: float | None
    model: str


@dataclass
class Claim:
    """One statement from generated text, checked against recorded observations."""

    text: str
    kind: Literal["object", "attribute", "action"]
    status: ClaimStatus
    evidence_ids: list[str]
    explanation: str


@dataclass
class ImageInfo:
    sha256: str
    original_size: list[int]
    processed_size: list[int]
    brightness: float
    sharpness: float


@dataclass
class QAEntry:
    question: str
    answer: str
    route: str
    evidence_ids: list[str]
    grounded: bool  # True if answered from recorded evidence, False if an unverified model guess


@dataclass
class AuditRecord:
    record_id: str
    timestamp: str
    image: ImageInfo
    models: dict[str, str]
    device: str
    detections: list[Detection] = field(default_factory=list)
    text_items: list[TextItem] = field(default_factory=list)
    relations: list[SpatialRelation] = field(default_factory=list)
    caption: Caption | None = None
    claims: list[Claim] = field(default_factory=list)
    consistency: float | None = None  # supported share of checkable claims; None if nothing was checkable
    uncertainty_flags: list[str] = field(default_factory=list)
    module_errors: dict[str, str] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    response: str = ""
    qa: list[QAEntry] = field(default_factory=list)
    notice: str = (
        "This record lists observable model outputs. It does not reveal the models' internal "
        "reasoning. A missing detection is not proof that an object is absent. Spatial relations "
        "are positions in the 2D image, not physical distances."
    )

    def to_dict(self) -> dict:
        return asdict(self)
