"""Includes the hallucination tests from report section 11."""

from conftest import det, text

from accessai.claims import consistency_score, extract_and_check


def status_of(claims, word):
    return next(c for c in claims if word in c.text).status


def test_object_with_confident_detection_is_supported():
    claims = extract_and_check("a dog on a couch", [det("D1", "dog"), det("D2", "couch")], [])
    assert status_of(claims, "dog") == "supported"
    assert status_of(claims, "couch") == "supported"
    assert next(c for c in claims if "dog" in c.text).evidence_ids == ["D1"]


def test_synonyms_map_to_coco_classes():
    claims = extract_and_check("a man on a sofa", [det("D1", "person"), det("D2", "couch")], [])
    assert status_of(claims, "man") == "supported"
    assert status_of(claims, "sofa") == "supported"


def test_specialized_personal_item_names_map_to_detector_labels():
    claims = extract_and_check("a pen beside earpods", [det("S1", "pen"), det("S2", "earbud")], [])
    assert status_of(claims, "pen") == "supported"
    assert status_of(claims, "earpods") == "supported"


def test_plural_nouns_match():
    claims = extract_and_check("people and dogs", [det("D1", "person"), det("D2", "dog")], [])
    assert all(c.status == "supported" for c in claims)


def test_hallucinated_object_is_not_confirmed_rather_than_declared_absent():
    claims = extract_and_check("a giraffe in a kitchen", [det("D1", "oven")], [])
    giraffe = next(c for c in claims if "giraffe" in c.text)
    assert giraffe.status == "not_confirmed"
    assert giraffe.evidence_ids == []
    assert "does not prove" in giraffe.explanation


def test_low_confidence_detection_is_weak():
    claims = extract_and_check("a cat", [det("D1", "cat", confidence=0.3)], [])
    assert status_of(claims, "cat") == "weak"


def test_noun_outside_detector_vocabulary_is_unverifiable():
    claims = extract_and_check("a brick wall", [], [])
    assert status_of(claims, "wall") == "unverifiable"


def test_count_disagreement_is_weak():
    claims = extract_and_check("two dogs", [det("D1", "dog")], [])
    assert status_of(claims, "dogs") == "weak"


def test_matching_count_is_supported():
    claims = extract_and_check("two dogs", [det("D1", "dog"), det("D2", "dog")], [])
    assert status_of(claims, "dogs") == "supported"


def test_compound_terms_match_as_one_class():
    claims = extract_and_check("a stop sign", [det("D1", "stop sign")], [])
    assert status_of(claims, "stop sign") == "supported"


def test_ocr_text_supports_a_claim():
    claims = extract_and_check("a menu on the wall", [], [text("T1", "Lunch MENU")])
    menu = next(c for c in claims if "menu" in c.text)
    assert menu.status == "supported"
    assert menu.evidence_ids == ["T1"]


def test_actions_and_attributes_are_marked_as_inference():
    claims = extract_and_check("a red car driving", [det("D1", "car")], [])
    by_kind = {c.kind: c for c in claims}
    assert by_kind["attribute"].status == "inference"
    assert by_kind["action"].status == "inference"


def test_meta_nouns_are_not_claims():
    assert extract_and_check("a picture of a group of dogs", [det("D1", "dog")], []) and all(
        "picture" not in c.text and "group" not in c.text
        for c in extract_and_check("a picture of a group of dogs", [det("D1", "dog")], [])
    )


def test_consistency_score_ignores_uncheckable_claims():
    claims = extract_and_check("a dog near a giraffe by a wall", [det("D1", "dog")], [])
    assert consistency_score(claims) == 0.5


def test_consistency_is_none_when_nothing_is_checkable():
    assert consistency_score(extract_and_check("a wall", [], [])) is None
