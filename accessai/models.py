"""Wrappers around the pretrained models (report sections 5.3, 5.4 and 5.6).

Each model loads on first use and is then reused. Heavy imports stay inside the
functions so the rest of the package and its tests do not need them.
"""

from __future__ import annotations

import threading

import numpy as np
from PIL import Image

from .config import DEFAULT_CONFIG, MODELS_DIR, Config
from .evidence import Caption, Detection, TextItem
from .spatial import horizontal_region


def select_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class ObjectDetector:
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.device = select_device()
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from ultralytics import YOLO

                MODELS_DIR.mkdir(exist_ok=True)
                self._model = YOLO(str(MODELS_DIR / self.config.yolo_weights))
        return self._model

    def predict_raw(self, rgb: np.ndarray, conf: float):
        # Ultralytics treats numpy input as BGR.
        return self._load().predict(rgb[:, :, ::-1], conf=conf, device=self.device, verbose=False)[0]

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        result = self.predict_raw(rgb, self.config.detection_min_confidence)
        width = rgb.shape[1]
        rows = sorted(
            zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()),
            key=lambda row: row[1],
            reverse=True,
        )
        return [
            Detection(
                id=f"D{i}",
                label=result.names[int(cls)],
                confidence=round(conf, 3),
                bbox=[round(v, 1) for v in box],
                region=horizontal_region(box, width),
            )
            for i, (cls, conf, box) in enumerate(rows, start=1)
        ]


class SmallItemDetector:
    """Targeted detector for personal items outside the COCO label set."""

    LABELS = ("a pen", "an earbud", "an airpod")

    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.device = select_device()
        self._processor = None
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from transformers import Owlv2ForObjectDetection, Owlv2Processor

                self._processor = Owlv2Processor.from_pretrained(self.config.specialized_model)
                self._model = Owlv2ForObjectDetection.from_pretrained(self.config.specialized_model)
                self._model.to(self.device)
        return self._processor, self._model

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        import torch

        processor, model = self._load()
        image = Image.fromarray(rgb)
        queries = [list(self.LABELS)]
        inputs = processor(text=queries, images=image, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.Tensor([image.size[::-1]])
        result = processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=self.config.specialized_min_confidence,
            target_sizes=target_sizes,
            text_labels=queries,
        )[0]
        candidates = []
        for score, box, label in zip(result["scores"], result["boxes"], result["text_labels"]):
            canonical = "pen" if "pen" in label.lower() else "earbud"
            candidates.append((canonical, float(score), box.tolist()))
        candidates.sort(key=lambda row: row[1], reverse=True)

        detections = []
        for label, confidence, box in candidates:
            if any(
                existing.label == label and _overlap_ratio(existing.bbox, box) > 0.5
                for existing in detections
            ):
                continue
            detections.append(
                Detection(
                    id=f"S{len(detections) + 1}",
                    label=label,
                    confidence=round(confidence, 3),
                    bbox=[round(value, 1) for value in box],
                    region=horizontal_region(box, rgb.shape[1]),
                )
            )
        return detections


def _overlap_ratio(first: list[float], second: list[float]) -> float:
    """Return intersection over the smaller box area for nested predictions."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if not intersection:
        return 0.0
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(min(first_area, second_area), 1e-9)


class _Blip:
    """Shared loading for the BLIP caption and question-answering models."""

    model_class = ""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = select_device()
        self._processor = None
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                import transformers

                self._processor = transformers.BlipProcessor.from_pretrained(self.model_name)
                model = getattr(transformers, self.model_class).from_pretrained(self.model_name)
                self._model = model.to(self.device).eval()
        return self._processor, self._model


class ImageCaptioner(_Blip):
    model_class = "BlipForConditionalGeneration"

    def __init__(self, config: Config = DEFAULT_CONFIG):
        super().__init__(config.caption_model)

    def caption(self, image: Image.Image) -> Caption:
        import torch

        processor, model = self._load()
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=30,
                num_beams=3,
                output_scores=True,
                return_dict_in_generate=True,
            )
        text = processor.decode(output.sequences[0], skip_special_tokens=True).strip()
        # sequences_scores is the beam's mean log probability per token.
        scores = getattr(output, "sequences_scores", None)
        confidence = round(float(torch.exp(scores[0])), 3) if scores is not None else None
        return Caption(text=text, confidence=confidence, model=self.model_name)


class QuestionAnswerer(_Blip):
    model_class = "BlipForQuestionAnswering"

    def __init__(self, config: Config = DEFAULT_CONFIG):
        super().__init__(config.vqa_model)

    def answer(self, image: Image.Image, question: str) -> str:
        import torch

        processor, model = self._load()
        inputs = processor(images=image, text=question, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=20)
        return processor.decode(output[0], skip_special_tokens=True).strip()


class TextReader:
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self._reader = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._reader is None:
                import easyocr

                self._reader = easyocr.Reader(
                    list(self.config.ocr_languages), gpu=select_device() == "cuda", verbose=False
                )
        return self._reader

    def read(self, rgb: np.ndarray) -> list[TextItem]:
        items = []
        for corners, text, confidence in self._load().readtext(rgb):
            text = text.strip()
            if not text or confidence < self.config.ocr_min_confidence:
                continue
            xs, ys = zip(*corners)
            items.append(
                TextItem(
                    id=f"T{len(items) + 1}",
                    text=text,
                    confidence=round(float(confidence), 3),
                    bbox=[round(float(min(xs)), 1), round(float(min(ys)), 1),
                          round(float(max(xs)), 1), round(float(max(ys)), 1)],
                )
            )
        return items
