import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from accessai.evidence import AuditRecord, Caption, Detection, ImageInfo, TextItem  # noqa: E402


def det(id_, label, confidence=0.9, bbox=(0, 0, 10, 10), region="center"):
    return Detection(id=id_, label=label, confidence=confidence, bbox=list(bbox), region=region)


def text(id_, value, confidence=0.9):
    return TextItem(id=id_, text=value, confidence=confidence, bbox=[0, 0, 10, 10])


@pytest.fixture
def make_record():
    def _make(detections=(), text_items=(), caption="a test scene", errors=None, flags=()):
        return AuditRecord(
            record_id="test",
            timestamp="2026-11-06T00:00:00+00:00",
            image=ImageInfo("hash", [100, 100], [100, 100], 120.0, 200.0),
            models={},
            device="cpu",
            detections=list(detections),
            text_items=list(text_items),
            caption=Caption(caption, 0.5, "stub"),
            module_errors=errors or {},
            uncertainty_flags=list(flags),
        )

    return _make
