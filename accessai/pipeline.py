"""The end-to-end analysis pipeline (report section 6)."""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from .claims import consistency_score, extract_and_check
from .config import DEFAULT_CONFIG, Config
from .evidence import AuditRecord
from .models import ImageCaptioner, ObjectDetector, QuestionAnswerer, SmallItemDetector, TextReader, select_device
from .preprocessing import PreparedImage, prepare_image
from .response import build_response
from .spatial import analyze_spatial


def _timed(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    return result, round((time.perf_counter() - start) * 1000, 1)


class AccessAIPipeline:
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.detector = ObjectDetector(config)
        self.small_item_detector = SmallItemDetector(config)
        self.captioner = ImageCaptioner(config)
        self.reader = TextReader(config)
        self.vqa = QuestionAnswerer(config)

    def analyze_bytes(self, data: bytes) -> tuple[PreparedImage, AuditRecord]:
        prepared, preprocess_ms = _timed(prepare_image, data, self.config)
        record = self.analyze_prepared(prepared)
        record.timings_ms["preprocessing"] = preprocess_ms
        record.timings_ms["total"] = round(record.timings_ms["total"] + preprocess_ms, 1)
        return prepared, record

    def analyze_prepared(self, prepared: PreparedImage) -> AuditRecord:
        cfg = self.config
        start = time.perf_counter()
        record = AuditRecord(
            record_id=uuid.uuid4().hex[:12],
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            image=prepared.info,
            models={
                "detection": cfg.yolo_weights,
                "small_item_detection": cfg.specialized_model,
                "caption": cfg.caption_model,
                "ocr": f"easyocr ({','.join(cfg.ocr_languages)})",
            },
            device=select_device(),
            uncertainty_flags=list(prepared.flags),
        )

        tasks = {
            "detection": (self.detector.detect, prepared.rgb),
            "small_item_detection": (self.small_item_detector.detect, prepared.rgb),
            "caption": (self.captioner.caption, prepared.pil),
            "ocr": (self.reader.read, prepared.rgb),
        }
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {name: pool.submit(_timed, fn, arg) for name, (fn, arg) in tasks.items()}

        # One failed module must not cost the user the output of the others.
        results = {}
        for name, future in futures.items():
            try:
                results[name], record.timings_ms[name] = future.result()
            except Exception as exc:  # noqa: BLE001
                record.module_errors[name] = f"{type(exc).__name__}: {exc}"

        record.detections = results.get("detection", []) + results.get("small_item_detection", [])
        record.caption = results.get("caption")
        record.text_items = results.get("ocr", [])
        record.relations = analyze_spatial(record.detections, cfg)

        if record.caption:
            evidence_start = time.perf_counter()
            # Without detector output nothing can be checked, so a detector failure must not
            # make every object in the caption look like a hallucination.
            if "detection" in record.module_errors:
                record.claims = []
            else:
                record.claims = extract_and_check(
                    record.caption.text, record.detections, record.text_items, cfg
                )
            record.timings_ms["claim_check"] = round((time.perf_counter() - evidence_start) * 1000, 1)
        record.consistency = consistency_score(record.claims)

        record.uncertainty_flags.extend(self._evidence_flags(record))
        record.response = build_response(record, cfg)
        record.timings_ms["total"] = round((time.perf_counter() - start) * 1000, 1)
        return record

    def _evidence_flags(self, record: AuditRecord) -> list[str]:
        cfg = self.config
        flags = []
        if record.module_errors:
            flags.append("module_failure")
        if "detection" not in record.module_errors and not record.detections:
            flags.append("no_detections")
        if any(d.confidence < cfg.detection_high_confidence for d in record.detections):
            flags.append("low_confidence_detections")
        if any(t.confidence < cfg.ocr_high_confidence for t in record.text_items):
            flags.append("low_confidence_text")
        caption = record.caption
        if caption and caption.confidence is not None and caption.confidence < cfg.caption_low_confidence:
            flags.append("low_confidence_caption")
        if any(c.status == "not_confirmed" for c in record.claims):
            flags.append("unconfirmed_claims")
        return flags
