"""Question routing (report section 5.9).

Routing is deterministic so that safety questions can never reach a generative model. Answers
come from the recorded evidence. Only questions the evidence cannot address go to BLIP VQA, and
those answers are labelled as unverified.
"""

from __future__ import annotations

import re
from typing import Callable

from .claims import extract_and_check
from .config import DEFAULT_CONFIG, Config
from .evidence import AuditRecord, Detection, QAEntry
from .response import build_response, count_phrase, join_list, pluralize, region_phrase, summarize_detections
from .vocabulary import TERM_TO_CLASSES, find_term

SAFETY_ANSWER = (
    "I can't tell you whether it is safe or whether a path is clear. A photo can't establish that, "
    "and I can't detect every obstacle. Please rely on your cane, guide dog, or a person you trust."
)
DISTANCE_ANSWER = (
    "I can't measure distance from a single photo. I can only say where things are in the picture."
)
MEDICAL_ANSWER = (
    "I can't give medical interpretation or dosing advice. Please check with a pharmacist or "
    "doctor."
)

_ROUTES = [
    ("safety", r"\b(safe|safely|danger|dangerous|hazard|obstacle|obstacles|clear path|path clear|"
               r"way clear|is it clear|cross|crossing|walk|step|trip|fall)\b"),
    ("medical", r"\b(medicine|medication|pill|pills|dose|dosage|diagnos\w*|symptom\w*|rash|"
                r"prescription|treatment)\b"),
    ("distance", r"\b(how far|how close|distance|how near|meters?|metres?|feet)\b"),
    ("count", r"\bhow many\b"),
    ("location", r"\bwhere\b"),
    ("text", r"\b(read|say|says|written|write|text|words?|letters?|label|document|sign say)\b"),
    ("presence", r"\b(is there|are there|do you see|can you see|any)\b"),
    ("objects", r"\b(what objects|which objects|what things|list)\b"),
    ("describe", r"\b(describe|what do you see|what is in|what's in|summary|summar\w+|"
                 r"around me|in front of me)\b"),
]


def route_question(question: str) -> str:
    q = question.lower()
    for name, pattern in _ROUTES:
        if re.search(pattern, q):
            return name
    return "vqa"


class Agent:
    def __init__(
        self,
        record: AuditRecord,
        vqa_fn: Callable[[str], str] | None = None,
        config: Config = DEFAULT_CONFIG,
    ):
        self.record = record
        self.vqa_fn = vqa_fn
        self.config = config

    def ask(self, question: str) -> QAEntry:
        route = route_question(question)
        entry = getattr(self, f"_{route}")(question)
        self.record.qa.append(entry)
        return entry

    # --- helpers ---

    def _entry(self, question, answer, route, ids=(), grounded=True):
        return QAEntry(question, answer, route, list(ids), grounded)

    def _visible_objects_sentence(self) -> str:
        if self._detector_unavailable():
            return "Object detection failed on this image, so I have no object information."
        phrases = summarize_detections(self.record.detections)
        if not phrases:
            return "No objects were detected in this image."
        return f"In this image I detected {join_list(phrases)}."

    def _matching(self, question: str) -> tuple[str | None, list[Detection]]:
        term = find_term(question)
        if term is None:
            return None, []
        classes = TERM_TO_CLASSES[term]
        return term, [d for d in self.record.detections if d.label in classes]

    def _detector_unavailable(self) -> bool:
        return "detection" in self.record.module_errors

    def _low_confidence_note(self, matches: list[Detection]) -> str:
        if any(d.confidence < self.config.detection_high_confidence for d in matches):
            return " Some of these detections are low confidence."
        return ""

    # --- routes ---

    def _safety(self, q):
        return self._entry(q, f"{SAFETY_ANSWER} {self._visible_objects_sentence()}", "safety",
                           [d.id for d in self.record.detections])

    def _medical(self, q):
        answer = MEDICAL_ANSWER
        if self.record.text_items:
            answer += " The visible text reads: " + ". ".join(t.text for t in self.record.text_items) + "."
        return self._entry(q, answer, "medical", [t.id for t in self.record.text_items])

    def _distance(self, q):
        _, matches = self._matching(q)
        answer = DISTANCE_ANSWER
        if matches:
            answer += f" I see {join_list(summarize_detections(matches))}."
        return self._entry(q, answer, "distance", [m.id for m in matches])

    def _count(self, q):
        if self._detector_unavailable():
            return self._entry(q, "Object detection failed on this image, so I can't count.", "count")
        term, matches = self._matching(q)
        if term is None:
            return self._vqa(q)
        if not matches:
            answer = (f"I didn't detect any {pluralize(term, 2)}, but some may be present "
                      "and not recognised.")
        else:
            answer = (f"I detected {count_phrase(matches[0].label, len(matches))}. "
                      f"The true number may differ.{self._low_confidence_note(matches)}")
        return self._entry(q, answer, "count", [m.id for m in matches])

    def _location(self, q):
        if self._detector_unavailable():
            return self._entry(q, "Object detection failed on this image, so I can't locate objects.", "location")
        term, matches = self._matching(q)
        if term is None:
            return self._vqa(q)
        if not matches:
            return self._entry(
                q, f"I didn't detect a {term}. It may still be there and not recognised.", "location"
            )
        ids = {m.id for m in matches}
        relations = [r for r in self.record.relations if r.subject_id in ids or r.object_id in ids]
        where = join_list([f"a {m.label} {region_phrase(m.region)}" for m in matches])
        answer = f"I see {where}."
        spoken = list(dict.fromkeys(_spoken_relation(r.description) for r in relations))[:3]
        if spoken:
            answer += " In the picture: " + ", ".join(spoken) + "."
        answer += " These are positions in the picture, not real distances."
        answer += self._low_confidence_note(matches)
        return self._entry(q, answer, "location", [m.id for m in matches])

    def _text(self, q):
        if "ocr" in self.record.module_errors:
            return self._entry(q, "Text reading failed on this image, so I can't read it.", "text")
        items = self.record.text_items
        if not items:
            return self._entry(
                q, "I didn't find readable text. If there is text, it may be blurred, "
                   "angled, or low contrast.", "text",
            )
        answer = "The visible text reads: " + ". ".join(t.text for t in items) + "."
        if any(t.confidence < self.config.ocr_high_confidence for t in items):
            answer += " Some of it may be misread."
        return self._entry(q, answer, "text", [t.id for t in items])

    def _presence(self, q):
        if self._detector_unavailable():
            return self._entry(q, "Object detection failed on this image, so I can't check that.", "presence")
        term, matches = self._matching(q)
        if term is None:
            return self._vqa(q)
        if matches:
            where = join_list(summarize_detections(matches))
            answer = f"Yes, I detected {where}.{self._low_confidence_note(matches)}"
        else:
            answer = (f"I didn't detect a {term}, but that doesn't prove there isn't one. "
                      f"{self._visible_objects_sentence()}")
        return self._entry(q, answer, "presence", [m.id for m in matches])

    def _objects(self, q):
        return self._entry(q, self._visible_objects_sentence(), "objects",
                           [d.id for d in self.record.detections])

    def _describe(self, q):
        return self._entry(q, build_response(self.record, self.config), "describe",
                           [d.id for d in self.record.detections] + [t.id for t in self.record.text_items])

    def _vqa(self, q):
        if self.vqa_fn is None:
            return self._entry(q, "I don't have enough evidence from this image to answer that.",
                               "insufficient_evidence")
        try:
            guess = self.vqa_fn(q)
        except Exception:  # noqa: BLE001
            return self._entry(q, "I couldn't answer that because the question model failed.",
                               "insufficient_evidence")
        answer = f"My best guess, which I can't verify from the evidence, is: {guess}."
        claims = extract_and_check(guess, self.record.detections, self.record.text_items, self.config)
        ids = [i for c in claims if c.status == "supported" for i in c.evidence_ids]
        return self._entry(q, answer, "vqa", ids, grounded=False)


def _spoken_relation(description: str) -> str:
    """Remove evidence ids and the trailing qualifier from text meant to be read aloud."""
    return re.sub(r" \((?:D|T)\d+\)", "", description).removesuffix(" in the image")
