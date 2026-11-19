import pytest
from conftest import det, text

from accessai.agent import Agent, route_question


def forbid_vqa(_question):
    raise AssertionError("This question must not reach the generative model")


@pytest.mark.parametrize(
    "question, route",
    [
        ("Is it safe to cross the road?", "safety"),
        ("Is the path clear?", "safety"),
        ("Are there any obstacles?", "safety"),
        ("How much of this medicine should I take?", "medical"),
        ("How far is the chair?", "distance"),
        ("How many people are there?", "count"),
        ("Where is the dog?", "location"),
        ("What does the sign say?", "text"),
        ("Read this label", "text"),
        ("Is there a laptop?", "presence"),
        ("List the objects", "objects"),
        ("Describe the scene", "describe"),
        ("What color is the car?", "vqa"),
    ],
)
def test_routing(question, route):
    assert route_question(question) == route


@pytest.mark.parametrize(
    "question",
    ["Is it safe to cross?", "Is the path clear?", "Can I walk forward?", "Is it dangerous here?"],
)
def test_safety_questions_never_affirm_and_never_use_the_model(make_record, question):
    agent = Agent(make_record([det("D1", "chair")]), forbid_vqa)
    entry = agent.ask(question)
    assert entry.route == "safety"
    assert "can't tell you" in entry.answer
    assert "cane" in entry.answer


def test_medical_questions_read_text_but_do_not_interpret(make_record):
    agent = Agent(make_record(text_items=[text("T1", "Take 2 daily")]), forbid_vqa)
    entry = agent.ask("What dose of this medicine should I take?")
    assert "can't give medical" in entry.answer
    assert "Take 2 daily" in entry.answer


def test_distance_is_refused_but_position_is_given(make_record):
    agent = Agent(make_record([det("D1", "chair", region="left")]), forbid_vqa)
    answer = agent.ask("How far is the chair?").answer
    assert "can't measure distance" in answer
    assert "on the left" in answer


def test_count_uses_detections_and_synonyms(make_record):
    agent = Agent(make_record([det("D1", "person"), det("D2", "person")]), forbid_vqa)
    entry = agent.ask("How many men are there?")
    assert "two people" in entry.answer
    assert entry.evidence_ids == ["D1", "D2"]


def test_absent_detection_is_not_reported_as_absence(make_record):
    agent = Agent(make_record([det("D1", "chair")]), forbid_vqa)
    answer = agent.ask("Is there a laptop?").answer
    assert "doesn't prove" in answer
    assert "chair" in answer


def test_low_confidence_presence_is_flagged(make_record):
    agent = Agent(make_record([det("D1", "cat", confidence=0.3)]), forbid_vqa)
    assert "low confidence" in agent.ask("Is there a cat?").answer


def test_text_route_handles_no_text(make_record):
    assert "didn't find readable text" in Agent(make_record(), forbid_vqa).ask("Read it").answer


def test_failed_detector_is_not_reported_as_nothing_detected(make_record):
    record = make_record(errors={"detection": "RuntimeError: boom"})
    agent = Agent(record, forbid_vqa)
    assert "failed" in agent.ask("Is there a dog?").answer
    assert "failed" in agent.ask("Is it safe?").answer


def test_model_answers_are_labelled_unverified(make_record):
    entry = Agent(make_record(), lambda q: "blue").ask("What color is the sky?")
    assert entry.route == "vqa"
    assert entry.grounded is False
    assert "can't verify" in entry.answer


def test_without_any_model_it_reports_insufficient_evidence(make_record):
    entry = Agent(make_record(), None).ask("What color is the sky?")
    assert entry.route == "insufficient_evidence"


def test_failing_model_does_not_crash(make_record):
    def broken(_question):
        raise RuntimeError("out of memory")

    assert Agent(make_record(), broken).ask("What color?").route == "insufficient_evidence"


def test_questions_are_added_to_the_audit_record(make_record):
    record = make_record()
    Agent(record, forbid_vqa).ask("Is it safe?")
    assert record.qa[0].question == "Is it safe?"
