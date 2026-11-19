# PATENT INVENTION DISCLOSURE & TECHNICAL REPORT
## AccessAI: Assistive Multi-Modal Vision System with Deterministic Claim Verification, Hallucination Auditing, and Safety-Gated Interaction

---

### CONFIDENTIAL ATTORNEY-CLIENT PRIVILEGED / INVENTOR DISCLOSURE DOCUMENT

* **Document Type:** Patent Technical Disclosure Specification & Evaluation Dossier
* **Project Name:** AccessAI
* **Repository Path:** `d:\access_ai\Access AI`
* **Date of Preparation:** November 17, 2026
* **Target Patent Filings:** USPTO Non-Provisional / Provisional Patent Application, PCT International Application, EPO Patent Application
* **Suggested Title of Invention:** *SYSTEM AND METHOD FOR DETERMINISTIC MULTI-MODAL EVIDENCE VERIFICATION, HALLUCINATION AUDITING, AND SAFETY-GATED CONVERSATIONAL ASSISTANCE IN ASSISTIVE VISION SYSTEMS*

---

## 1. TECHNICAL FIELD OF THE INVENTION

The present invention relates generally to computerized image analysis, computer vision (CV), natural language processing (NLP), machine learning, and assistive sensory aids. More specifically, the invention relates to:
1. Deconstructing generative multi-modal image descriptions into discrete atomic claims;
2. Deterministically verifying generative claims against independent sensory bounding boxes and optical character recognition tokens without internal neural network introspection;
3. Quantifying claim consistency and isolating hallucinations using a five-tier epistemic status taxonomy;
4. Intercepting and enforcing deterministic safety gates on interactive conversational queries regarding physical navigation, obstacle clearance, metric distance, and medical instructions; and
5. Establishing a privacy-preserving, dual-consent, immutable audit trail for assistive sensory perceptions.

### Relevant Patent Classifications
* **G06V 20/00 / G06V 20/50:** Scene analysis / spatial scene understanding.
* **G06V 10/70 / G06V 10/82:** Pattern recognition, deep convolutional/transformer neural networks.
* **G06F 40/20 / G06F 40/30:** Natural language processing, syntactic parsing, semantic extraction.
* **G06N 3/00 / G06N 20/00:** Artificial intelligence, multimodal foundation models, machine learning.
* **G09B 21/00:** Visual aids and assistive apparatuses for blind or visually impaired persons.
* **G08B 21/02:** Safety monitoring responsive to hazard detection and alert generation.

---

## 2. BACKGROUND OF THE INVENTION & PRIOR ART DEFICIENCIES

### 2.1 Technological Context
Visually impaired individuals increasingly rely on artificial intelligence agents to interpret their physical environments. Recent advances in Vision-Language Models (VLMs) and Multimodal Large Language Models (MLLMs)—such as BLIP, LLaVA, GPT-4V, and Google Gemini—enable end-to-end generation of natural language captions and answers to open-ended visual questions.

### 2.2 The Technical Problem: Generative Hallucination and Safety Hazards
Despite high linguistic fluency, generative models suffer from fundamental vulnerabilities that render them dangerous when deployed as primary sensory aids:
1. **Pervasive Latent Hallucination:** Generative language decoders generate tokens autoregressively based on probability distributions $P(w_t \mid w_{<t}, I)$. When visual signals are ambiguous, occluded, or low contrast, linguistic co-occurrence priors dominate. The model invents plausible but non-existent physical objects (e.g., claiming a crosswalk exists, generating non-existent vehicles, or inventing obstacles).
2. **Black-Box Evidentiary Opacity:** End-to-end models cannot output verifiable evidentiary provenance. A user or auditor cannot verify which pixel cluster, bounding box, or sensory observation caused a specific word to appear in a description.
3. **Conversational Sycophancy & Unsafe Extrapolations:** When visually impaired users ask safety-critical questions (e.g., *"Is it safe to cross the road?"*, *"Is the path clear?"*, *"How far away is the vehicle?"*, or *"What dose of this medicine should I take?"*), conventional generative agents frequently answer affirmatively or guess distances from 2D monocular images, creating severe physical risks.
4. **"Absence as Proof" Fallacy:** Existing systems conflate a model's failure to mention an object with proof that the object is absent from the physical scene, misleading users regarding unseen hazards.
5. **Privacy Vulnerabilities:** Many assistive apps stream uncompressed user images and OCR streams (containing private mail, medical prescriptions, and biometric data) to third-party cloud servers without granular user consent.

---

## 3. SUMMARY OF THE INVENTION

The present invention solves these technical problems through an asynchronous, multi-branch, fault-tolerant perception and deterministic claim-verification architecture:
1. **Parallel Heterogeneous Perception:** Decouples vision-language caption generation from empirical feature extraction (a primary high-speed YOLO11s detector, a specialized zero-shot OWLv2 micro-item detector with nested-area suppression, and an EasyOCR engine).
2. **Deterministic Syntactic Claim Deconstruction:** Generative text is parsed into discrete semantic noun chunks, numeric quantifiers, and verb/adjective predicates using dependency parsing.
3. **Ontology-Mapped Bounding Box Verification:** Each noun chunk is mapped via a multi-word vocabulary ontology onto detectable class sets. Detections and OCR text are queried to assign each claim to one of five deterministic epistemic categories: `supported`, `weak`, `not_confirmed`, `unverifiable`, and `inference`.
4. **Quantitative Evidence Consistency Metric:** The system computes an objective consistency ratio $C = |\text{Supported}| / |\text{Checkable}|$, excluding uncheckable and inferential statements.
5. **Deterministic Safety-Gated Router:** A rule-based lexical interceptor intercepts user questions prior to any generative model invocation. Physical safety, navigation, metric distance, and medical questions receive pre-compiled non-affirmative refusals grounded strictly in confirmed sensory evidence, completely eliminating generative hallucinations from safety-critical paths.
6. **Dual-Consent Relational Audit Store:** System metadata and raw image bytes are decoupled into separate user consent tiers with automatic time-to-live (TTL) data destruction.

---

## 4. DETAILED DESCRIPTION OF PREFERRED EMBODIMENTS & ARCHITECTURE

### 4.1 Input Preprocessing and Signal Quality Assessment
Implemented in `accessai/preprocessing.py`:
* **EXIF Normalization:** Automatically transposes orientation flags (`ImageOps.exif_transpose`) ensuring coordinate alignment.
* **Aspect-Preserving Scaling:** Clamps resolution to a maximum dimension of 1280 pixels and validates a minimum threshold of 32 pixels.
* **Photometric Luminance:** Computes mean greyscale pixel intensity $\mu_I = \frac{1}{HW} \sum I(x, y)$. If $\mu_I < 50.0$, sets a `low_light` flag.
* **Laplacian Variance Focus Metric:** Evaluates edge focus using the discrete 2D Laplacian operator $\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$:
  $$\sigma^2_L = \text{Var}(\nabla^2 I)$$
  If $\sigma^2_L < 60.0$, sets a `blurry` flag.

### 4.2 Parallel Heterogeneous Feature Extraction Pipeline
Implemented in `accessai/models.py` and `accessai/pipeline.py`:
* Handled asynchronously across a multi-worker `ThreadPoolExecutor`.
* **Primary Detector Branch:** YOLO11s identifies common objects across 80 classes with minimum threshold $\tau = 0.25$.
* **Targeted Micro-Item Detector Branch:** OWLv2 transformer executed exclusively with targeted prompts: $Q = \{\text{"a pen"}, \text{"an earbud"}, \text{"an airpod"}\}$.
  * **Nested Box Suppression Ratio:** Evaluates candidate overlap using:
    $$O(A, B) = \frac{|A \cap B|}{\min(|A|, |B|)}$$
    If $O(A, B) > 0.5$, suppresses duplicate nested predictions.
* **Text Reading Branch:** EasyOCR outputs text strings, detection polygons, and confidence scores.
* **Generative Caption Branch:** BLIP conditional generator produces text with 3-beam search and sequence log-probability score calculation.
* **Fault Isolation:** If any individual module raises an exception, the remaining branches continue execution without system failure (`record.module_errors`).

### 4.3 2D Axis-Dominance Spatial Modeling
Implemented in `accessai/spatial.py`:
* Evaluates top confident detections pairwise.
* Computes bounding box Intersection over Union (IoU). If $\text{IoU} \ge 0.10$, relationship is defined as `"overlaps"`.
* If non-overlapping, calculates centroid deltas $\Delta x$ and $\Delta y$. The axis exhibiting greater absolute separation determines the primary directional descriptor: `"left of"`, `"right of"`, `"above"`, or `"below"`.
* Outputs qualified statements explicitly designating relationships as 2D image coordinates rather than 3D physical distances.

### 4.4 Deterministic Claim Extraction & Verification Engine
Implemented in `accessai/claims.py`:
* Deconstructs caption text using SpaCy dependency parsing (`en_core_web_sm`).
* Excludes meta-nouns (`META_NOUNS`), pronouns, and non-claim stopwords.
* Resolves compound multi-word entities (e.g., "stop sign", "wine glass").
* Extracts numeric modifiers (`dep_ == "nummod"`).
* Categorizes each claim into the five-tier taxonomy:
  1. `supported`: Confident detection ($\ge 0.50$) matching class and stated count, or exact OCR match.
  2. `weak`: Detection confidence between $0.25$ and $0.50$, or stated count disagrees with detected count.
  3. `not_confirmed`: Object class is within detector ontology, but zero detections exist (hallucination candidate).
  4. `unverifiable`: Object class is outside detector ontology.
  5. `inference`: Adjectives and action verbs representing model interpretation.
* Computes evidence consistency score:
  $$C = \frac{|\text{Supported}|}{|\text{Supported}| + |\text{Weak}| + |\text{Not Confirmed}|}$$

### 4.5 Deterministic Safety Router
Implemented in `accessai/agent.py`:
* Evaluates user input against pre-compiled regex intent rules:
  * **Safety / Obstacles / Crossing:** Returns pre-compiled refusal and appends confirmed detections.
  * **Medical / Dosing:** Returns refusal and reads visible text verbatim.
  * **Distance / Depth:** Refuses 3D measurement and returns 2D image region.
  * **Count / Location / Presence:** Synthesizes answer strictly from structured evidence records.
  * **Fallback (VQA):** Only out-of-scope visual questions reach BLIP VQA, prepended with an uncertainty disclaimer and verified against the claim engine.

### 4.6 Dual-Consent Audit Storage
Implemented in `accessai/audit.py`:
* Persists structured records into SQLite (`audit.sqlite3`).
* Raw image bytes written to disk only with explicit secondary user consent.
* Automatic purging of records exceeding retention threshold (30 days).

---

## 5. EXPERIMENTAL RESULTS & WORKING EXAMPLES

Empirical evaluation conducted on 400 test images from Microsoft COCO val2017:

### 5.1 Verification Accuracy Metrics
* **Total Ground Truth Objects:** 2,738
* **Supported Claims Accuracy (Precision):** **98.16%** (427 / 435 confirmed physically present in ground truth).
* **Weak Claims Precision:** **89.80%** (44 / 49 confirmed present).
* **Not-Confirmed Claims True Absence:** **71.43%** (35 / 49 confirmed true generative hallucinations).
* **Mean Evidence Consistency Score:** **82.98%**.

### 5.2 Real-Time Latency Breakdown
* Preprocessing: 8.8 ms
* Object Detection (YOLO11s): 247.1 ms
* Caption Generation (BLIP): 565.0 ms
* Optical Character Recognition: 169.4 ms
* **Claim Verification Overhead:** **5.1 ms** (Median: 4.0 ms, P95: 9.0 ms)
* **Total End-to-End Pipeline Latency:** **573.9 ms** (Median: 514.2 ms)

---

## 6. PATENT CLAIMS (STATUTORY CLAIM SET)

### Claim 1 (Independent Method Claim - Verification Engine)
A computer-implemented method for verifying generative image descriptions in an assistive vision apparatus, the method comprising:
1. receiving, by at least one processor, an input image captured by an optical sensor;
2. processing the input image across a plurality of parallel asynchronous feature extraction branches, wherein:
   * a first branch executes an object detection model to generate a set of empirical object detections comprising bounding box coordinates, class labels, and detection confidence values;
   * a second branch executes an optical character recognition model to extract a set of recognized text items with spatial bounding boxes and text confidence scores; and
   * a third branch executes a generative vision-language model to synthesize a natural language description of the input image;
3. decomposing, by natural language parsing executed by the at least one processor, the synthesized natural language description into a plurality of discrete semantic claims;
4. for each respective semantic claim of the plurality of discrete semantic claims:
   * extracting a candidate entity term and an associated numerical quantifier if present;
   * mapping the candidate entity term to a canonical class set using a domain-specific vocabulary ontology;
   * evaluating the canonical class set against the set of empirical object detections and the set of recognized text items; and
   * assigning to the respective semantic claim an epistemic verification status selected from a predetermined taxonomy comprising at least a supported status, a weak status, and an unconfirmed status;
5. computing an aggregate evidence consistency score representing a ratio of supported claims relative to total checkable claims; and
6. synthesizing an output speech representation of the scene, wherein assertions classified under the unconfirmed status are verbally segregated from assertions classified under the supported status.

### Claim 2 (Dependent Claim - Epistemic Status Criteria)
The method of claim 1, wherein assigning the epistemic verification status comprises:
* assigning the supported status when at least one empirical object detection matches the canonical class set with a confidence exceeding a predetermined firm threshold and matches the associated numerical quantifier;
* assigning the weak status when an empirical object detection matches the canonical class set with a confidence falling below the predetermined firm threshold, or when an empirical detection count disagrees with the associated numerical quantifier;
* assigning the unconfirmed status when the canonical class set exists in a detectable vocabulary of the object detection model, but zero empirical object detections match the canonical class set; and
* assigning an unverifiable status when the candidate entity term is absent from the detectable vocabulary of the object detection model and absent from the set of recognized text items.

### Claim 3 (Dependent Claim - Inference Segregation)
The method of claim 1, wherein decomposing the synthesized natural language description further comprises identifying non-auxiliary verbs and descriptive adjectives within the description and categorizing each identified verb and adjective as an unverified model inference.

### Claim 4 (Dependent Claim - Micro-Item Suppression)
The method of claim 1, wherein the first branch further executes a secondary open-vocabulary zero-shot object detector restricted to a targeted query vocabulary comprising personal utility micro-items, wherein candidate micro-item detections are filtered against detections from the object detection model using an intersection-over-smaller-area ratio:
$$\frac{|A \cap B|}{\min(|A|, |B|)} > \theta$$
wherein $\theta$ is a predetermined threshold, thereby suppressing redundant nested predictions.

### Claim 5 (Independent Method Claim - Deterministic Safety-Gated Conversational Assistance)
A computer-implemented method for safety-gated conversational assistance for visually impaired users, the method comprising:
1. maintaining, in a computer memory, an evidence record comprising a plurality of verified object detections, optical character recognition tokens, and pairwise 2D spatial relationships generated from an image;
2. receiving an interactive conversational query from a user;
3. parsing the query with a deterministic pattern-matching router prior to any invocation of a generative language model;
4. in response to the query matching a predetermined safety hazard pattern or path clearance pattern:
   * blocking execution of any generative machine learning model; and
   * generating a deterministic safety refusal message indicating an inability to confirm pathway safety, combined with an enumeration of objects confirmed present in the evidence record;
5. in response to the query matching a medical or pharmaceutical dosing pattern:
   * blocking execution of any generative machine learning model; and
   * outputting a medical disclaimer combined verbatim with recognized text tokens from the evidence record without diagnostic interpretation; and
6. in response to the query requesting factual presence, count, or location of an object:
   * synthesizing an answer deterministically and exclusively from the evidence record.

### Claim 6 (Dependent Claim - Fallback Quarantine)
The method of claim 5, further comprising:
* in response to the query failing to match any predetermined safety, medical, or factual pattern:
  * transmitting the query and the image to a visual question-answering neural network;
  * receiving a generated answer;
  * prepending an explicit verbal disclaimer designating the generated answer as an unverified estimation; and
  * validating candidate entities extracted from the generated answer against the verified object detections in the evidence record.

### Claim 7 (Independent System Claim - Assistive Vision Apparatus)
An assistive vision apparatus comprising:
* an image sensor configured to capture an image of an environment;
* an audio output interface;
* a memory storing executable instructions and an evidence audit database; and
* one or more processors configured by the executable instructions to perform operations comprising:
  * preprocessing the captured image and computing image luminance and Laplacian variance blur metrics;
  * executing in parallel an object detector, an optical character reader, and an autoregressive caption generator;
  * parsing a caption generated by the autoregressive caption generator into discrete entity chunks;
  * deterministically cross-referencing each discrete entity chunk against bounding boxes produced by the object detector and the optical character reader;
  * classifying each chunk into an epistemic category indicating whether the chunk is confirmed by physical evidence, represents a count mismatch, or represents an unconfirmed hallucination candidate;
  * converting the classified chunks into an audio response via the audio output interface, wherein unconfirmed chunks are explicitly audibly flagged; and
  * recording the classified chunks, bounding boxes, and image quality metrics into the evidence audit database under a user-configured data retention policy.

### Claim 8 (Dependent Claim - Decoupled Privacy Consent)
The apparatus of claim 7, wherein recording into the evidence audit database comprises:
* persisting symbolic metadata of the evidence audit database in response to a first user consent authorization; and
* persisting raw image pixel data of the captured image only in response to a second, independent user consent authorization distinct from the first user consent authorization.

---

## 7. PRIOR ART DIFFERENTIATION & LEGAL STRATEGY

### 7.1 Differentiation Table

| Prior Art Category | Representative Systems | Key Limitations | AccessAI Technological Solution |
| :--- | :--- | :--- | :--- |
| **Point Solution Assistive Apps** | Microsoft Seeing AI, Google Lookout | Isolated modes; no cross-verification; black-box captions. | Heterogeneous parallel pipeline; caption validated against simultaneous object & text detections. |
| **Generative Multimodal LLMs** | Be My Eyes with GPT-4V, Envision | Autoregressive hallucinations; sycophantic safety answers; no localized bounding-box provenance. | Deterministic SpaCy-based claim parsing; 5-tier status classification; 98.16% precision on supported claims. |
| **Conversational Chatbots** | ChatGPT Vision, Gemini | Vulnerable to prompt injection; estimates dangerous 3D metric distances from 2D images. | Hardware/software deterministic regex router physically preventing safety/medical queries from reaching generative models. |
| **Academic Hallucination Benchmarks** | POPE, CHAIR, Woodpecker | Post-hoc offline evaluation; relies on secondary LLMs (which also hallucinate); slow latency. | Real-time lightweight deterministic verification running in **5.1 ms**; zero secondary LLM dependency. |

### 7.2 Strategy for Overcoming 35 U.S.C. § 101 Rejections
* Under the USPTO *Alice/Mayo* framework, patent examiners often issue § 101 rejections for software inventions claiming "abstract mathematical algorithms" or "mental processes".
* **Overcoming Arguments:**
  1. **Concrete Technical Problem in Computer Vision:** The claims address the specific technical failure mode of generative neural network hallucination in vision-language models.
  2. **Physical Transformation of Optical Sensor Signals:** The system ingests optical sensor bytes, performs physical quality assessment, and coordinates parallel specialized inference engines.
  3. **Inventive Step Beyond Generic Computers:** The integration of the nested-area micro-item suppression algorithm ($O(A,B)$), the five-tier epistemic state machine, and the deterministic safety router provides an inventive technological combination that improves computer vision system reliability.

---

*End of Disclosure Document.*
