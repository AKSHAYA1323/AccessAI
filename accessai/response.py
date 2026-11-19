"""Response generation: a concise spoken summary that keeps observation and inference apart."""

from __future__ import annotations

from collections import defaultdict

from .config import DEFAULT_CONFIG, Config
from .evidence import AuditRecord, Detection

FLAG_MESSAGES = {
    "low_light": "The image is dark, so results may be less reliable.",
    "blurry": "The image looks blurry, so results may be less reliable.",
    "low_resolution": "The image is low resolution, so small details may be missed.",
    "module_failure": "Part of the analysis failed, so this description is incomplete.",
}

_NUMBERS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
_IRREGULAR = {"person": "people", "skis": "skis", "sheep": "sheep", "knife": "knives"}


def pluralize(label: str, count: int) -> str:
    if count == 1:
        return label
    if label in _IRREGULAR:
        return _IRREGULAR[label]
    return label + ("es" if label.endswith(("s", "x", "ch", "sh")) else "s")


def count_phrase(label: str, count: int) -> str:
    amount = _NUMBERS[count] if count < len(_NUMBERS) else str(count)
    return f"{amount} {pluralize(label, count)}"


def join_list(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def region_phrase(region: str) -> str:
    return "in the center" if region == "center" else f"on the {region}"


def summarize_detections(detections: list[Detection]) -> list[str]:
    """Group boxes by label, e.g. 'two people on the left and one in the center' becomes one phrase."""
    grouped: dict[str, list[Detection]] = defaultdict(list)
    for det in detections:
        grouped[det.label].append(det)

    phrases = []
    for label, items in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        regions = {d.region for d in items}
        if len(regions) == 1:
            phrases.append(f"{count_phrase(label, len(items))} {region_phrase(items[0].region)}")
        else:
            phrases.append(f"{count_phrase(label, len(items))} across the image")
    return phrases


def build_response(record: AuditRecord, config: Config = DEFAULT_CONFIG) -> str:
    parts: list[str] = []

    if record.caption and record.caption.text:
        parts.append(f"AI description, which may be inaccurate: {record.caption.text}.")

    firm = [d for d in record.detections if d.confidence >= config.detection_high_confidence]
    weak = [d for d in record.detections if d.confidence < config.detection_high_confidence]
    if firm:
        parts.append(f"Detected: {join_list(summarize_detections(firm))}.")
    if weak:
        labels = join_list(sorted({d.label for d in weak}))
        parts.append(f"Possibly also, with low confidence: {labels}.")
    if "detection" not in record.module_errors and not record.detections:
        parts.append("No objects were detected, but objects may still be present.")

    if record.text_items:
        readable = [t for t in record.text_items if t.confidence >= config.ocr_high_confidence]
        unclear = [t for t in record.text_items if t.confidence < config.ocr_high_confidence]
        if readable:
            parts.append("Visible text: " + ". ".join(t.text for t in readable) + ".")
        if unclear:
            parts.append("Text that may be misread: " + ". ".join(t.text for t in unclear) + ".")

    unconfirmed = [c.text for c in record.claims if c.status == "not_confirmed"]
    if unconfirmed:
        parts.append(
            f"The description mentions {join_list(unconfirmed)}, but this could not be confirmed."
        )

    parts.extend(FLAG_MESSAGES[f] for f in record.uncertainty_flags if f in FLAG_MESSAGES)
    return " ".join(parts) or "No information could be extracted from this image."
