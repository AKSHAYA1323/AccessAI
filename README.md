# AccessAI

An assistive agent for visually impaired users. It describes a photo aloud, reads visible text,
answers questions about the image, and records the evidence behind what it says.

Alongside each description it checks the generated statements against what the object detector and
OCR actually found, then flags anything unsupported. It explains observable model outputs, not the
models' internal reasoning.

> AccessAI is an information tool. It cannot tell you whether a route is clear or safe, and it does
> not replace a white cane, guide dog, or other mobility aid.

## Setup

Requires Python 3.12 (other 3.x versions are untested). A GPU is recommended; the CPU works but is
slower.

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

Install PyTorch. For an NVIDIA GPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Or for CPU only:

```bash
pip install torch torchvision
```

Then install everything else:

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Model weights (about 3 GB in total) download on first use, so the first analysis is slow. To use a
phone, open the app's Network URL and upload a photo; the phone's file picker can open its camera.
Live camera capture in a phone browser needs HTTPS.

## How it works

Image input → Preprocessing → Detection / Captioning / OCR / Spatial analysis → Evidence aggregation
→ Claim checking and audit → Response generation → Voice and visual output

| Report section | Code |
| --- | --- |
| 5.1, 5.2 Acquisition and preprocessing | `app.py`, `accessai/preprocessing.py` |
| 5.3, 5.4, 5.6 Detection (YOLO11 + targeted OWLv2), captioning (BLIP), OCR (EasyOCR) | `accessai/models.py` |
| 5.5 Spatial relationships | `accessai/spatial.py` |
| 5.7 Explainability and audit | `accessai/claims.py`, `accessai/evidence.py`, `accessai/audit.py` |
| 5.8 Text to speech | `accessai/tts.py` |
| 5.9 Interactive agent | `accessai/agent.py` |
| 6 Pipeline and response generation | `accessai/pipeline.py`, `accessai/response.py` |
| 11 Performance evaluation | `evaluation/` |

All thresholds are in `accessai/config.py`.

### Claim statuses

| Status | Meaning |
| --- | --- |
| Supported | A confident detection or OCR result backs the claim. |
| Weak | Evidence is low confidence, or a stated count disagrees with the detector. |
| Not confirmed | The detector could have found this but did not. A possible hallucination or a missed detection, never reported as proof of absence. |
| Cannot be checked | Outside the detector's 80 COCO categories, so there is no evidence either way. |
| AI interpretation | An action or attribute, which these models record no evidence for. |

**Evidence consistency** is the share of checkable claims (supported, weak, not confirmed) that are
supported.

### Question routing

Routing is rule-based, so safety questions can never reach a generative model.

| Question type | Handled by |
| --- | --- |
| Safety, clear path | Fixed refusal, plus a list of what was detected |
| Medical | Fixed refusal, plus the visible text read verbatim |
| Distance | Fixed refusal, plus position in the picture |
| Count, location, presence, list of objects | Detection and spatial evidence |
| Reading text | OCR evidence |
| Anything else | BLIP question answering, labelled as an unverified guess |

### Privacy

- Nothing is saved unless the user ticks **Save audit records**. Images need a second, separate tick.
- Saved records are deleted after 30 days, and can be deleted sooner under **Audit history**.
- Records may contain text read from images, such as names or addresses.
- Offline speech is the default. The gTTS option sends the spoken text to Google.
- Models run locally. The only network use is downloading weights, and gTTS if selected.

## Tests

```bash
python -m pytest tests
```

`tests/test_integration.py` uses the real models. Add `--ignore=tests/test_integration.py` for a
run of a few seconds.

## Evaluation

```bash
python evaluation/prepare_coco.py --dev 100 --test 400
```

```bash
python evaluation/evaluate.py --split test
```

Results are saved in `evaluation/results/`. METEOR needs Java and is reported as `null` without it.

Method notes, which should be stated alongside any reported results:

- The models are pretrained and were not trained here. All images come from COCO val2017, sampled
  with a fixed seed, and the dev and test sets share no images.
- Captions use simple lowercase tokenization rather than the Java PTB tokenizer.
- CIDEr's document frequencies come from the evaluated sample rather than the full dataset, and
  the plain CIDEr variant is used. Scores are not directly comparable to published CIDEr-D values.
- Occluded means at least one pair of annotated boxes has IoU above 0.3. Cluttered means 10 or more
  annotated objects. An image can be in both groups. Ordinary means neither.
- Low light is simulated by reducing brightness to 20%. It is not real low-light photography.
- OCR is measured on rendered text images, because COCO has no text annotations. Scores will be
  higher than on real photos.
- Claim validation uses COCO annotations as ground truth, which only covers objects within the 80
  categories. Synonym mapping is approximate (for example, "table" maps to "dining table").

COCO's license and terms should be checked before redistributing any images.

## Known limitations

- YOLO11 only knows 80 COCO categories; a targeted OWLv2 stage additionally checks for pens
  and earbuds, which are common personal items outside COCO.
- Claim checking confirms that an object exists. It does not verify relationships, such as whether
  a person is really "riding" a horse.
- BLIP and YOLO can both be wrong, so agreement between them does not guarantee a claim is true.
- Spatial relations are 2D image positions. They are never physical distances.
- Questions are typed. Speech input is not implemented.
- OCR is English only by default (`ocr_languages` in `accessai/config.py`).
- Ultralytics is licensed under AGPL-3.0, which matters if this is ever distributed commercially.

## Possible next steps

See section 14 of the project report. Testing with visually impaired users is the most important
one, following appropriate ethical review.
