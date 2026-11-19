"""Image acquisition and preprocessing (report sections 5.1 and 5.2)."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import DEFAULT_CONFIG, Config
from .evidence import ImageInfo


class InvalidImageError(ValueError):
    """The input could not be used as an image."""


@dataclass
class PreparedImage:
    rgb: np.ndarray  # HxWx3 uint8, the array every model and bounding box refers to
    pil: Image.Image
    info: ImageInfo
    flags: list[str]


def prepare_image(data: bytes, config: Config = DEFAULT_CONFIG) -> PreparedImage:
    """Validate raw image bytes, fix orientation, resize, and measure quality."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("This file could not be read as an image.") from exc

    # Phone photos often store their rotation in EXIF rather than in the pixels.
    image = ImageOps.exif_transpose(image).convert("RGB")
    original_size = list(image.size)

    if min(image.size) < config.min_side:
        raise InvalidImageError(
            f"This image is too small to analyse (minimum {config.min_side} pixels per side)."
        )

    if max(image.size) > config.max_side:
        image.thumbnail((config.max_side, config.max_side), Image.Resampling.LANCZOS)

    rgb = np.asarray(image)
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(grey.mean())
    sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())

    flags = []
    if brightness < config.low_light_brightness:
        flags.append("low_light")
    if sharpness < config.blur_laplacian_variance:
        flags.append("blurry")
    if min(original_size) < config.low_resolution_side:
        flags.append("low_resolution")

    info = ImageInfo(
        sha256=hashlib.sha256(data).hexdigest(),
        original_size=original_size,
        processed_size=list(image.size),
        brightness=round(brightness, 1),
        sharpness=round(sharpness, 1),
    )
    return PreparedImage(rgb=rgb, pil=image, info=info, flags=flags)


def draw_boxes(prepared_rgb: np.ndarray, detections: list, text_items: list) -> np.ndarray:
    """Return a copy of the image with detection and OCR boxes drawn on it."""
    canvas = prepared_rgb.copy()
    scale = max(canvas.shape[:2]) / 1000
    thickness = max(2, round(2 * scale))
    font_scale = max(0.5, 0.6 * scale)

    def draw(bbox, label, colour):
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)
        (w, h), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        top = max(y1, h + base + 2)
        cv2.rectangle(canvas, (x1, top - h - base - 2), (x1 + w + 2, top), colour, -1)
        cv2.putText(
            canvas, label, (x1 + 1, top - base), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )

    for det in detections:
        draw(det.bbox, f"{det.id} {det.label} {det.confidence:.2f}", (0, 90, 200))
    for item in text_items:
        draw(item.bbox, f"{item.id} text", (170, 40, 120))
    return canvas
