"""Claim-to-evidence checking (report section 5.7): the project's principal contribution.

Generated text is split into claims, and each claim gets one status:

  supported      A confident detection or OCR result backs it.
  weak           Evidence exists but is low confidence, or a stated count disagrees.
  not_confirmed  The detector could have found this but did not. A possible hallucination,
                 or a missed detection. Never reported as proof of absence.
  unverifiable   Outside what the detector knows, so there is no evidence either way.
  inference      An action or attribute. These models record no evidence for them.
"""

from __future__ import annotations

import re
import threading

from .config import DEFAULT_CONFIG, Config
from .evidence import Claim, Detection, TextItem
from .vocabulary import META_NOUNS, NUMBER_WORDS, TERM_TO_CLASSES

_NLP = None
_NLP_LOCK = threading.Lock()

_NON_CLAIM_VERBS = {"be", "have", "do", "appear", "seem", "show", "look"}
_NON_CLAIM_ADJECTIVES = {"other", "many", "several", "few", "some", "such", "same"}


def _nlp():
    global _NLP
    with _NLP_LOCK:
        if _NLP is None:
            import spacy

            _NLP = spacy.load("en_core_web_sm", disable=["ner"])
    return _NLP


def _noun_term(chunk) -> str:
    """The longest vocabulary match ending at the chunk's head, else the head's lemma."""
    head = chunk.root
    lemma = head.lemma_.lower()
    previous = chunk.doc[head.i - 1] if head.i > chunk.start else None
    if previous is not None:
        for second in (lemma, head.text.lower()):
            compound = f"{previous.text.lower()} {second}"
            if compound in TERM_TO_CLASSES:
                return compound
    if lemma in TERM_TO_CLASSES:
        return lemma
    return head.text.lower() if head.text.lower() in TERM_TO_CLASSES else lemma


def _stated_count(chunk) -> int | None:
    for token in chunk:
        if token.dep_ == "nummod":
            word = token.text.lower()
            if word.isdigit():
                return int(word)
            if word in NUMBER_WORDS:
                return NUMBER_WORDS[word]
    return None


def _check_object(
    phrase: str,
    term: str,
    count: int | None,
    detections: list[Detection],
    text_items: list[TextItem],
    config: Config,
) -> Claim:
    classes = TERM_TO_CLASSES.get(term)
    matches = [d for d in detections if classes and d.label in classes]
    ocr_matches = [t for t in text_items if re.search(rf"\b{re.escape(term)}\b", t.text, re.I)]
    firm = [d for d in matches if d.confidence >= config.detection_high_confidence]

    def claim(status, ids, explanation):
        return Claim(text=phrase, kind="object", status=status, evidence_ids=ids, explanation=explanation)

    if firm:
        ids = [d.id for d in firm]
        if count is not None and count != len(matches):
            return claim(
                "weak", ids,
                f"The text states {count}, but the detector found {len(matches)}. "
                "The object is supported; the count is not.",
            )
        best = max(d.confidence for d in firm)
        return claim("supported", ids, f"Detected as {firm[0].label} (best confidence {best:.2f}).")
    if ocr_matches:
        return claim(
            "supported", [t.id for t in ocr_matches], f"The word '{term}' appears in the OCR text."
        )
    if matches:
        best = max(d.confidence for d in matches)
        return claim(
            "weak", [d.id for d in matches],
            f"Detected as {matches[0].label}, but only at low confidence ({best:.2f}).",
        )
    if classes:
        return claim(
            "not_confirmed", [],
            f"The detector can recognise {' or '.join(classes)} but found none. This may be a "
            "hallucination or a missed detection; it does not prove the object is absent.",
        )
    return claim(
        "unverifiable", [],
        f"'{term}' is outside the detector's categories, so there is no evidence either way.",
    )


def extract_and_check(
    text: str,
    detections: list[Detection],
    text_items: list[TextItem],
    config: Config = DEFAULT_CONFIG,
) -> list[Claim]:
    """Split generated text into claims and check each against recorded observations."""
    doc = _nlp()(text)
    claims: list[Claim] = []
    seen: set[str] = set()

    for chunk in doc.noun_chunks:
        if chunk.root.pos_ == "PRON":
            continue
        term = _noun_term(chunk)
        if term in META_NOUNS or term in seen:
            continue
        seen.add(term)
        phrase = " ".join(t.text for t in chunk if t.pos_ != "DET")
        claims.append(_check_object(phrase, term, _stated_count(chunk), detections, text_items, config))

    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in seen or token.is_stop and token.pos_ != "VERB":
            continue
        if token.pos_ == "VERB" and lemma not in _NON_CLAIM_VERBS:
            kind, label = "action", "An action"
        elif token.pos_ == "ADJ" and lemma not in _NON_CLAIM_ADJECTIVES:
            kind, label = "attribute", "An attribute"
        else:
            continue
        seen.add(lemma)
        claims.append(
            Claim(
                text=token.text.lower(),
                kind=kind,
                status="inference",
                evidence_ids=[],
                explanation=f"{label} interpreted by the model. Not independently verified.",
            )
        )
    return claims


def consistency_score(claims: list[Claim]) -> float | None:
    """Share of checkable claims that are supported. None if no claim could be checked."""
    checkable = [c for c in claims if c.status in ("supported", "weak", "not_confirmed")]
    if not checkable:
        return None
    return round(sum(c.status == "supported" for c in checkable) / len(checkable), 3)
