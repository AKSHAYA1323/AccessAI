"""Central settings. Every threshold used by the pipeline is defined here."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


@dataclass(frozen=True)
class Config:
    # Models
    yolo_weights: str = "yolo11s.pt"
    specialized_model: str = "google/owlv2-base-patch16-ensemble"
    specialized_min_confidence: float = 0.20
    caption_model: str = "Salesforce/blip-image-captioning-base"
    vqa_model: str = "Salesforce/blip-vqa-base"
    ocr_languages: tuple[str, ...] = ("en",)

    # Preprocessing
    max_side: int = 1280
    min_side: int = 32
    low_light_brightness: float = 50.0  # mean grey level, 0-255
    blur_laplacian_variance: float = 60.0
    low_resolution_side: int = 320

    # Evidence thresholds
    detection_min_confidence: float = 0.25  # below this a box is discarded
    detection_high_confidence: float = 0.50  # at or above this a box counts as firm evidence
    ocr_min_confidence: float = 0.30
    ocr_high_confidence: float = 0.60
    caption_low_confidence: float = 0.30

    # Spatial analysis
    overlap_iou: float = 0.10
    max_spatial_objects: int = 6

    # Audit storage
    audit_db: Path = DATA_DIR / "audit.sqlite3"
    audit_images_dir: Path = DATA_DIR / "audit_images"
    retention_days: int = 30


DEFAULT_CONFIG = Config()
