"""Download COCO val2017 annotations and a fixed random sample of its images.

The models are pretrained and not trained here, so COCO train2017 is not needed. Sampled images
are split by image id into a dev set and a held-out test set, so no image appears in both.

Usage: python evaluation/prepare_coco.py --dev 100 --test 400
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from accessai.config import DATA_DIR  # noqa: E402

COCO_DIR = DATA_DIR / "coco"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_URL = "http://images.cocodataset.org/val2017/{}"
SEED = 2017


def download_annotations() -> None:
    wanted = ["annotations/instances_val2017.json", "annotations/captions_val2017.json"]
    if all((COCO_DIR / name).exists() for name in wanted):
        return
    archive = COCO_DIR / "annotations.zip"
    with requests.get(ANNOTATIONS_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(archive, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="annotations") as bar:
            for chunk in response.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    with zipfile.ZipFile(archive) as z:
        for name in wanted:
            z.extract(name, COCO_DIR)
    archive.unlink()


def download_image(file_name: str) -> None:
    path = COCO_DIR / "val2017" / file_name
    if path.exists():
        return
    response = requests.get(IMAGE_URL.format(file_name), timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=int, default=100)
    parser.add_argument("--test", type=int, default=400)
    args = parser.parse_args()

    (COCO_DIR / "val2017").mkdir(parents=True, exist_ok=True)
    download_annotations()

    with open(COCO_DIR / "annotations" / "instances_val2017.json", encoding="utf-8") as f:
        images = json.load(f)["images"]
    images.sort(key=lambda image: image["id"])
    random.Random(SEED).shuffle(images)
    sample = images[: args.dev + args.test]

    split = {
        "seed": SEED,
        "dev": [image["id"] for image in sample[: args.dev]],
        "test": [image["id"] for image in sample[args.dev:]],
    }
    (COCO_DIR / "split.json").write_text(json.dumps(split), encoding="utf-8")

    names = [image["file_name"] for image in sample]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(tqdm(pool.map(download_image, names), total=len(names), desc="images"))
    print(f"Ready: {args.dev} dev and {args.test} test images in {COCO_DIR}")


if __name__ == "__main__":
    main()
