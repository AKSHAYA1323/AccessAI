"""Runs the real models on a real photo. Slow on first run because weights are downloaded."""

from pathlib import Path

import pytest

from accessai.agent import Agent
from accessai.pipeline import AccessAIPipeline
from accessai.tts import synthesize_offline

ultralytics = pytest.importorskip("ultralytics")
BUS = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"


@pytest.fixture(scope="module")
def analysed():
    pipeline = AccessAIPipeline()
    prepared, record = pipeline.analyze_bytes(BUS.read_bytes())
    return pipeline, prepared, record


def test_all_modules_run(analysed):
    _, _, record = analysed
    assert record.module_errors == {}
    assert record.response
    assert record.timings_ms["total"] > 0


def test_detector_finds_the_bus_and_people(analysed):
    _, _, record = analysed
    labels = {d.label for d in record.detections}
    assert {"bus", "person"} <= labels


def test_caption_and_claims_exist(analysed):
    _, _, record = analysed
    assert record.caption.text
    assert record.claims


def test_agent_answers_from_evidence_and_with_the_model(analysed):
    pipeline, prepared, record = analysed
    agent = Agent(record, lambda q: pipeline.vqa.answer(prepared.pil, q))
    assert agent.ask("How many people are there?").grounded
    guess = agent.ask("What color is the bus?")
    assert guess.route == "vqa" and not guess.grounded and guess.answer


def test_offline_speech_produces_audio():
    audio, mime = synthesize_offline("This is a test of AccessAI.")
    assert mime == "audio/wav" and len(audio) > 1000
