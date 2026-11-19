"""Measure AccessAI against COCO (report section 11).

Metrics: detection precision, recall and mAP; caption BLEU, METEOR and CIDEr; response time;
evidence consistency, validated against COCO's human annotations; performance per scene condition
(ordinary, cluttered, occluded, low light); and OCR accuracy on rendered text images.

Usage: python evaluation/evaluate.py --split test
Results are written to evaluation/results/.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import io
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from accessai.claims import consistency_score, extract_and_check  # noqa: E402
from accessai.config import DATA_DIR, DEFAULT_CONFIG  # noqa: E402
from accessai.pipeline import AccessAIPipeline  # noqa: E402
from accessai.preprocessing import prepare_image  # noqa: E402
from accessai.spatial import iou  # noqa: E402
from accessai.vocabulary import TERM_TO_CLASSES, find_term  # noqa: E402

COCO_DIR = DATA_DIR / "coco"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CLUTTERED_MIN_OBJECTS = 10
OCCLUSION_IOU = 0.3
LOW_LIGHT_FACTOR = 0.2
MATCH_IOU = 0.5
METEOR_TIMEOUT_SECONDS = 300

OCR_SAMPLES = [
    "EXIT", "NO SMOKING", "Emergency Exit", "Room 204", "Open 9am to 5pm", "Wet Floor",
    "Pull to open", "Best before 12 March", "Platform 3", "Please wait here", "Milk 500 ml",
    "Library closed on Sunday",
]


def load_ground_truth(split: str):
    from pycocotools.coco import COCO

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(COCO_DIR / "annotations" / "instances_val2017.json"))
    ids = json.loads((COCO_DIR / "split.json").read_text(encoding="utf-8"))[split]
    captions_json = json.loads(
        (COCO_DIR / "annotations" / "captions_val2017.json").read_text(encoding="utf-8")
    )
    wanted = set(ids)
    captions: dict[int, list[str]] = {}
    for ann in captions_json["annotations"]:
        if ann["image_id"] in wanted:
            captions.setdefault(ann["image_id"], []).append(ann["caption"])
    return coco, ids, captions


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def ground_truth_boxes(coco, image_id):
    """Returns non-crowd boxes as (label, xyxy). Crowd regions are left out of P/R matching."""
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
    return [
        (coco.cats[a["category_id"]]["name"], xywh_to_xyxy(a["bbox"]))
        for a in anns
        if not a.get("iscrowd")
    ]


def match_counts(predictions, truths):
    """Greedy, same-class matching at IoU 0.5. predictions are (label, confidence, xyxy)."""
    unmatched = list(truths)
    true_positives = 0
    for label, _, box in sorted(predictions, key=lambda p: -p[1]):
        best, best_iou = None, MATCH_IOU
        for truth in unmatched:
            if truth[0] == label and (overlap := iou(box, truth[1])) >= best_iou:
                best, best_iou = truth, overlap
        if best is not None:
            unmatched.remove(best)
            true_positives += 1
    return true_positives, len(predictions) - true_positives, len(unmatched)


def precision_recall(tp, fp, fn):
    return {
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "ground_truth_objects": tp + fn,
    }


def scene_conditions(truths):
    occluded = any(iou(a[1], b[1]) > OCCLUSION_IOU for a, b in combinations(truths, 2))
    cluttered = len(truths) >= CLUTTERED_MIN_OBJECTS
    conditions = []
    if cluttered:
        conditions.append("cluttered")
    if occluded:
        conditions.append("occluded")
    if not conditions:
        conditions.append("ordinary")
    return conditions


def coco_map(coco, ids, pipeline, images):
    """COCO mAP. Requires a very low confidence threshold, as the metric integrates over all."""
    from pycocotools.cocoeval import COCOeval

    name_to_cat = {cat["name"]: cat_id for cat_id, cat in coco.cats.items()}
    results = []
    for image_id in tqdm(ids, desc="mAP"):
        output = pipeline.detector.predict_raw(images[image_id].rgb, conf=0.001)
        for cls, conf, box in zip(
            output.boxes.cls.tolist(), output.boxes.conf.tolist(), output.boxes.xyxy.tolist()
        ):
            x1, y1, x2, y2 = box
            results.append(
                {
                    "image_id": image_id,
                    "category_id": name_to_cat[output.names[int(cls)]],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": conf,
                }
            )
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator = COCOeval(coco, coco.loadRes(results), "bbox")
        evaluator.params.imgIds = ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {"mAP@.50:.95": round(float(evaluator.stats[0]), 4), "mAP@.50": round(float(evaluator.stats[1]), 4)}


def normalize_caption(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def caption_metrics(generated: dict[int, str], references: dict[int, list[str]]) -> dict:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider

    hypotheses = {i: [normalize_caption(c)] for i, c in generated.items()}
    refs = {i: [normalize_caption(c) for c in references[i]] for i in generated}

    with contextlib.redirect_stdout(io.StringIO()):
        bleu, _ = Bleu(4).compute_score(refs, hypotheses)
    cider, _ = Cider().compute_score(refs, hypotheses)
    metrics = {f"BLEU-{n + 1}": round(float(v), 4) for n, v in enumerate(bleu)}
    metrics["CIDEr"] = round(float(cider), 4)
    metrics["METEOR"] = meteor_score(refs, hypotheses)
    return metrics


def meteor_score(refs, hypotheses):
    """The official METEOR 1.5 jar, which needs Java. Returns None if it cannot run.

    The jar cannot find its data when its path contains a space, and pycocoevalcap's wrapper
    hangs when the jar exits early. So the jar runs from a copy on a space-free path, driven
    directly with a watchdog.
    """
    import pycocoevalcap.meteor.meteor as wrapper

    source = Path(wrapper.__file__).parent
    work = Path(tempfile.gettempdir()) / "accessai_meteor"
    if not (work / "data" / "paraphrase-en.gz").exists():
        shutil.copytree(source, work, dirs_exist_ok=True)

    process = subprocess.Popen(
        ["java", "-jar", "-Xmx2G", "meteor-1.5.jar", "-", "-", "-stdio", "-l", "en", "-norm"],
        cwd=work, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, encoding="utf-8",
    )
    watchdog = threading.Timer(METEOR_TIMEOUT_SECONDS, process.kill)
    watchdog.start()

    def exchange(line: str) -> str:
        process.stdin.write(line + "\n")
        process.stdin.flush()
        return process.stdout.readline().strip()

    try:
        stats = [
            exchange(" ||| ".join(["SCORE", *refs[i], hypotheses[i][0].replace("|||", "")]))
            for i in hypotheses
        ]
        exchange(" ||| ".join(["EVAL", *stats]))  # first output line: the first image's score
        for _ in range(len(stats) - 1):
            process.stdout.readline()
        return round(float(process.stdout.readline().strip()), 4)  # last line: the corpus score
    except (OSError, ValueError) as exc:
        print(f"METEOR skipped: {exc}", file=sys.stderr)
        return None
    finally:
        watchdog.cancel()
        process.kill()


def validate_claims(claims, truth_labels: set[str], tally: Counter) -> None:
    """Compare claim statuses with COCO's human annotations, for objects COCO can label."""
    for claim in claims:
        if claim.kind != "object" or claim.status not in ("supported", "weak", "not_confirmed"):
            continue
        term = find_term(claim.text)
        if term is None:
            continue
        in_truth = bool(set(TERM_TO_CLASSES[term]) & truth_labels)
        tally[f"{claim.status}:{'present' if in_truth else 'absent'}"] += 1


def share(tally: Counter, status: str, outcome: str):
    total = tally[f"{status}:present"] + tally[f"{status}:absent"]
    return round(tally[f"{status}:{outcome}"] / total, 4) if total else None


def darken(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    ImageEnhance.Brightness(image).enhance(LOW_LIGHT_FACTOR).save(buffer, format="PNG")
    return buffer.getvalue()


def ocr_accuracy(pipeline) -> dict:
    """Render known strings at easy and harder settings, then compare OCR output with the truth."""
    try:
        font = ImageFont.truetype("arial.ttf", 44)
    except OSError:
        font = ImageFont.load_default(44)

    def render(text, angle, foreground, background):
        image = Image.new("RGB", (900, 300), background)
        ImageDraw.Draw(image).text((60, 120), text, font=font, fill=foreground)
        return image.rotate(angle, expand=False, fillcolor=background)

    settings = {
        "clean": dict(angle=0, foreground=(0, 0, 0), background=(255, 255, 255)),
        "angled": dict(angle=10, foreground=(0, 0, 0), background=(255, 255, 255)),
        "low_contrast": dict(angle=0, foreground=(120, 120, 120), background=(160, 160, 160)),
    }
    results = {}
    for name, kwargs in settings.items():
        scores = []
        for sample in OCR_SAMPLES:
            items = pipeline.reader.read(np.asarray(render(sample, **kwargs)))
            found = " ".join(item.text for item in items)
            scores.append(difflib.SequenceMatcher(None, sample.lower(), found.lower()).ratio())
        results[name] = round(statistics.mean(scores), 4)
    return results


def summarize_times(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "mean_ms": round(statistics.mean(ordered), 1),
        "median_ms": round(statistics.median(ordered), 1),
        "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 1),
    }


def run(split: str, limit: int | None) -> dict:
    coco, ids, references = load_ground_truth(split)
    ids = ids[:limit] if limit else ids
    pipeline = AccessAIPipeline()

    images = {}
    for image_id in ids:
        path = COCO_DIR / "val2017" / coco.imgs[image_id]["file_name"]
        images[image_id] = prepare_image(path.read_bytes())

    generated: dict[int, str] = {}
    timings: dict[str, list[float]] = {}
    status_counts: Counter = Counter()
    validation: Counter = Counter()
    consistency: list[float] = []
    condition_counts: dict[str, list[int]] = {}
    condition_consistency: dict[str, list[float]] = {}
    condition_images: Counter = Counter()
    flag_counts: Counter = Counter()

    def add_condition(name, tp, fp, fn, score):
        totals = condition_counts.setdefault(name, [0, 0, 0])
        totals[0] += tp
        totals[1] += fp
        totals[2] += fn
        condition_images[name] += 1
        if score is not None:
            condition_consistency.setdefault(name, []).append(score)

    # Load all models first, so one-off loading time is not counted as response time.
    pipeline.analyze_prepared(images[ids[0]])

    overall = [0, 0, 0]
    for image_id in tqdm(ids, desc="pipeline"):
        prepared = images[image_id]
        record = pipeline.analyze_prepared(prepared)
        if record.module_errors:
            raise RuntimeError(f"Image {image_id}: {record.module_errors}")

        generated[image_id] = record.caption.text
        for stage, ms in record.timings_ms.items():
            timings.setdefault(stage, []).append(ms)
        status_counts.update(c.status for c in record.claims)
        flag_counts.update(record.uncertainty_flags)
        if record.consistency is not None:
            consistency.append(record.consistency)

        truths = ground_truth_boxes(coco, image_id)
        validate_claims(record.claims, {label for label, _ in truths}, validation)

        predictions = [(d.label, d.confidence, d.bbox) for d in record.detections]
        tp, fp, fn = match_counts(predictions, truths)
        overall = [overall[0] + tp, overall[1] + fp, overall[2] + fn]
        for condition in scene_conditions(truths):
            add_condition(condition, tp, fp, fn, record.consistency)

    # Same images, synthetically darkened. Quality flags are checked here too.
    low_light_flagged = 0
    for image_id in tqdm(ids, desc="low light"):
        dark = prepare_image(darken(images[image_id].pil))
        low_light_flagged += "low_light" in dark.flags
        detections = pipeline.detector.detect(dark.rgb)
        caption = pipeline.captioner.caption(dark.pil)
        claims = extract_and_check(caption.text, detections, [], DEFAULT_CONFIG)
        predictions = [(d.label, d.confidence, d.bbox) for d in detections]
        tp, fp, fn = match_counts(predictions, ground_truth_boxes(coco, image_id))
        add_condition("low_light", tp, fp, fn, consistency_score(claims))

    return {
        "split": split,
        "images": len(ids),
        "device": pipeline.detector.device,
        "models": {
            "detection": pipeline.config.yolo_weights,
            "caption": pipeline.config.caption_model,
        },
        "detection": {
            **precision_recall(*overall),
            "confidence_threshold": DEFAULT_CONFIG.detection_min_confidence,
            "match_iou": MATCH_IOU,
            **coco_map(coco, ids, pipeline, images),
        },
        "captions": caption_metrics(generated, references),
        "response_time": {stage: summarize_times(v) for stage, v in timings.items()},
        "evidence_consistency": {
            "mean_per_image": round(statistics.mean(consistency), 4) if consistency else None,
            "images_with_checkable_claims": len(consistency),
            "claim_status_counts": dict(status_counts),
            "validated_against_coco_annotations": {
                "supported_claims_truly_present": share(validation, "supported", "present"),
                "weak_claims_truly_present": share(validation, "weak", "present"),
                "not_confirmed_claims_truly_absent": share(validation, "not_confirmed", "absent"),
                "counts": dict(validation),
            },
        },
        "conditions": {
            name: {
                "images": condition_images[name],
                **precision_recall(*totals),
                "mean_consistency": round(statistics.mean(condition_consistency[name]), 4)
                if condition_consistency.get(name)
                else None,
            }
            for name, totals in condition_counts.items()
        },
        "low_light_flag_rate_on_darkened_images": round(low_light_flagged / len(ids), 4),
        "uncertainty_flag_counts": dict(flag_counts),
        "ocr_similarity_on_rendered_text": ocr_accuracy(pipeline),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--limit", type=int, help="Use only the first N images, for a quick check")
    args = parser.parse_args()

    results = run(args.split, args.limit)
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"metrics_{args.split}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
