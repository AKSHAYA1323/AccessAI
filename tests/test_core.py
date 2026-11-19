"""Preprocessing, spatial analysis, response text and audit storage."""

import io
from dataclasses import replace

import numpy as np
import pytest
from conftest import det, text
from PIL import Image

from accessai.audit import AuditStore
from accessai.config import DEFAULT_CONFIG
from accessai.evidence import Claim
from accessai.preprocessing import InvalidImageError, draw_boxes, prepare_image
from accessai.response import build_response, summarize_detections
from accessai.spatial import analyze_spatial, horizontal_region, iou, relation_between


def png(array) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array.astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def noise(height=400, width=400, low=0, high=255):
    return np.random.default_rng(0).integers(low, high, (height, width, 3))


# --- preprocessing ---


def test_rejects_non_image_data():
    with pytest.raises(InvalidImageError):
        prepare_image(b"this is not an image")


def test_rejects_tiny_images():
    with pytest.raises(InvalidImageError):
        prepare_image(png(noise(10, 10)))


def test_large_images_are_resized_and_aspect_ratio_kept():
    prepared = prepare_image(png(noise(1000, 2000)))
    assert prepared.info.processed_size == [1280, 640]
    assert prepared.info.original_size == [2000, 1000]


def test_exif_rotation_is_applied():
    buffer = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees
    Image.fromarray(noise(100, 200).astype(np.uint8)).save(buffer, format="JPEG", exif=exif)
    assert prepare_image(buffer.getvalue()).info.processed_size == [100, 200]


def test_dark_image_is_flagged():
    assert "low_light" in prepare_image(png(noise(low=0, high=20))).flags


def test_flat_image_is_flagged_as_blurry():
    assert "blurry" in prepare_image(png(np.full((400, 400, 3), 128))).flags


def test_sharp_bright_image_has_no_flags():
    assert prepare_image(png(noise())).flags == []


def test_draw_boxes_does_not_change_the_original():
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    drawn = draw_boxes(rgb, [det("D1", "dog", bbox=(10, 10, 50, 50))], [text("T1", "hi")])
    assert rgb.sum() == 0
    assert drawn.sum() > 0


# --- spatial ---


def test_iou_values():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ([0, 0, 10, 10], [50, 0, 60, 10], "left of"),
        ([50, 0, 60, 10], [0, 0, 10, 10], "right of"),
        ([0, 0, 10, 10], [0, 50, 10, 60], "above"),
        ([0, 50, 10, 60], [0, 0, 10, 10], "below"),
        ([0, 0, 10, 10], [1, 1, 11, 11], "overlaps"),
    ],
)
def test_relations(a, b, expected):
    assert relation_between(a, b, 0.1) == expected


def test_regions():
    assert horizontal_region([0, 0, 10, 10], 300) == "left"
    assert horizontal_region([140, 0, 160, 10], 300) == "center"
    assert horizontal_region([280, 0, 300, 10], 300) == "right"


def test_spatial_output_is_limited_and_says_image_plane():
    detections = [det(f"D{i}", "cup", bbox=(i * 20, 0, i * 20 + 10, 10)) for i in range(10)]
    relations = analyze_spatial(detections)
    assert len(relations) == 15  # the 6 most confident objects give 15 pairs
    assert all("in the image" in r.description for r in relations)


# --- response ---


def test_detection_summary_groups_by_label():
    detections = [det("D1", "person", region="left"), det("D2", "person", region="left")]
    assert summarize_detections(detections) == ["two people on the left"]


def test_response_separates_caption_from_detections_and_labels_it(make_record):
    response = build_response(make_record([det("D1", "dog")], caption="a dog outside"))
    assert "AI description, which may be inaccurate: a dog outside." in response
    assert "Detected: one dog in the center." in response


def test_response_reports_unconfirmed_claims_and_no_detections(make_record):
    record = make_record(caption="a giraffe")
    record.claims = [Claim("giraffe", "object", "not_confirmed", [], "")]
    response = build_response(record)
    assert "mentions giraffe, but this could not be confirmed" in response
    assert "objects may still be present" in response


def test_response_includes_quality_warnings(make_record):
    assert "dark" in build_response(make_record(flags=["low_light"]))


def test_response_does_not_say_no_objects_when_the_detector_failed(make_record):
    record = make_record(errors={"detection": "boom"}, flags=["module_failure"])
    response = build_response(record)
    assert "No objects were detected" not in response
    assert "incomplete" in response


def test_response_never_makes_safety_claims(make_record):
    response = build_response(make_record([det("D1", "chair")])).lower()
    assert "safe" not in response and "clear" not in response


# --- audit storage ---


@pytest.fixture
def store(tmp_path):
    return AuditStore(
        replace(DEFAULT_CONFIG, audit_db=tmp_path / "a.db", audit_images_dir=tmp_path / "imgs")
    )


def test_saved_record_round_trips(store, make_record):
    store.save(make_record([det("D1", "dog")]))
    assert store.get("test")["detections"][0]["label"] == "dog"


def test_image_is_only_stored_when_provided(store, make_record, tmp_path):
    store.save(make_record())
    assert not (tmp_path / "imgs").exists()
    store.save(make_record(), b"image-bytes")
    assert (tmp_path / "imgs" / "test.img").read_bytes() == b"image-bytes"


def test_resaving_without_image_keeps_the_image_link(store, make_record):
    store.save(make_record(), b"image-bytes")
    store.save(make_record())
    assert store.list_records()[0]["image_path"]


def test_delete_removes_record_and_image(store, make_record, tmp_path):
    store.save(make_record(), b"image-bytes")
    store.delete("test")
    assert store.get("test") is None
    assert not (tmp_path / "imgs" / "test.img").exists()


def test_records_past_retention_are_purged(store, make_record):
    record = make_record()
    record.timestamp = "2020-01-01T00:00:00+00:00"
    store.save(record)
    store.purge_expired()
    assert store.list_records() == []
