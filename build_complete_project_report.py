import os
from pathlib import Path

# Set matplotlib & ultralytics config paths
os.environ['MPLCONFIGDIR'] = str(Path('figures/.mpl').resolve())
os.environ['YOLO_CONFIG_DIR'] = str(Path('models').resolve())

import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_heading_1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(14)
    run.bold = True
    run.font.color.rgb = RGBColor(30, 58, 138)
    return p

def add_heading_2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(12)
    run.bold = True
    run.font.color.rgb = RGBColor(30, 64, 175)
    return p

def add_heading_3(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(11)
    run.bold = True
    run.font.color.rgb = RGBColor(55, 65, 81)
    return p

def add_p(doc, text, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=6):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.25
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)
    run.bold = bold
    run.italic = italic
    return p

def add_bullet(doc, bold_prefix, text):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.2
    if bold_prefix:
        r1 = p.add_run(bold_prefix)
        r1.font.name = 'Times New Roman'
        r1.font.size = Pt(11)
        r1.bold = True
    r2 = p.add_run(text)
    r2.font.name = 'Times New Roman'
    r2.font.size = Pt(11)
    return p

def add_figure_image(doc, img_path, caption_text, width_inches=5.8):
    if Path(img_path).exists():
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(10)
        p_img.paragraph_format.space_after = Pt(4)
        run_img = p_img.add_run()
        run_img.add_picture(str(img_path), width=Inches(width_inches))

        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(12)
        run_cap = p_cap.add_run(caption_text)
        run_cap.font.name = 'Arial'
        run_cap.font.size = Pt(9.5)
        run_cap.italic = True
        run_cap.font.color.rgb = RGBColor(71, 85, 105)

def create_table(doc, headers, data_rows, col_widths=None):
    num_cols = len(headers)
    tbl = doc.add_table(rows=len(data_rows) + 1, cols=num_cols)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False

    # Header row
    for c_idx, h in enumerate(headers):
        cell = tbl.cell(0, c_idx)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_background(cell, "E2E8F0")
        set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(h)
        r.font.name = 'Arial'
        r.font.size = Pt(9.5)
        r.bold = True
        r.font.color.rgb = RGBColor(15, 23, 42)

    # Data rows
    for r_idx, row in enumerate(data_rows):
        bg = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row):
            cell = tbl.cell(r_idx + 1, c_idx)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(str(val))
            r.font.name = 'Times New Roman'
            r.font.size = Pt(9.5)

    if col_widths:
        for row in tbl.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)

    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_after = Pt(6)

def generate_report():
    doc = Document()

    # Configure Margins: 1-inch on all sides
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)
        s.page_width = Inches(8.5)
        s.page_height = Inches(11.0)

    # ==========================================
    # COVER PAGE
    # ==========================================
    p_top = doc.add_paragraph()
    p_top.paragraph_format.space_before = Pt(36)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(6)
    r_t = p_title.add_run("AccessAI: Real-Time Assistive Vision with Deterministic Claim Verification and Hallucination Auditing")
    r_t.font.name = 'Arial'
    r_t.font.size = Pt(20)
    r_t.bold = True
    r_t.font.color.rgb = RGBColor(15, 23, 42)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(36)
    r_s = p_sub.add_run("A Multi-Branch Asynchronous Perception, Spatial Reasoning, and Safety-Gated Conversational Pipeline")
    r_s.font.name = 'Arial'
    r_s.font.size = Pt(13)
    r_s.font.color.rgb = RGBColor(71, 85, 105)

    # Divider line
    p_div = doc.add_paragraph()
    pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="12" w:space="1" w:color="334155"/></w:pBdr>')
    p_div._p.get_or_add_pPr().append(pBdr)

    # Submission Meta Table
    tbl_meta = doc.add_table(rows=8, cols=2)
    tbl_meta.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_info = [
        ("Submitted By :", "Akshaya"),
        ("Programme :", "B.Tech — Computer Science & Engineering (AI & ML)"),
        ("Project Domain :", "Computer Vision, Multimodal NLP, and Assistive Technology"),
        ("Core Architecture :", "YOLO11s + OWLv2 + BLIP + EasyOCR + Deterministic Claim Engine"),
        ("Application Interface :", "Streamlit Real-Time Dashboard with Offline/Online TTS"),
        ("Evaluation Dataset :", "Microsoft COCO val2017 (500 Samples: 100 Dev / 400 Test)"),
        ("Audit & Storage :", "SQLite Decoupled Dual-Consent Evidence Persistence"),
        ("Date :", "November 2026")
    ]
    for idx, (label, val) in enumerate(meta_info):
        c1, c2 = tbl_meta.cell(idx, 0), tbl_meta.cell(idx, 1)
        c1.width, c2.width = Inches(2.2), Inches(4.3)
        p1, p2 = c1.paragraphs[0], c2.paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p1.paragraph_format.space_after = Pt(3)
        p2.paragraph_format.space_after = Pt(3)
        r1 = p1.add_run(label)
        r1.font.name = 'Arial'
        r1.font.size = Pt(10.5)
        r1.bold = True
        r2 = p2.add_run(val)
        r2.font.name = 'Times New Roman'
        r2.font.size = Pt(10.5)

    p_div2 = doc.add_paragraph()
    p_div2.paragraph_format.space_before = Pt(12)
    pBdr2 = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="12" w:space="1" w:color="334155"/></w:pBdr>')
    p_div2._p.get_or_add_pPr().append(pBdr2)

    doc.add_page_break()

    # ==========================================
    # ABSTRACT
    # ==========================================
    add_heading_1(doc, "Abstract")
    add_p(doc,
        "Assistive artificial intelligence systems for visually impaired individuals hold immense potential to foster daily independence. "
        "However, modern generative Vision-Language Models (VLMs) and Multimodal Large Language Models (MLLMs) suffer from stochastic hallucinations, "
        "generating plausible yet fabricated assertions regarding physical obstacles, object presences, counts, and spatial relationships. In safety-critical "
        "assistive contexts, unverified generative descriptions and conversational answers pose severe physical hazards (e.g., misjudging street crossings, "
        "misreading medical dosage, or misidentifying pathway obstructions). To resolve these deficiencies, this project presents AccessAI, an explainable, "
        "multi-modal assistive vision architecture that decouples generative scene description from independent sensory evidence verification.")
    add_p(doc,
        "The AccessAI framework executes an asynchronous, fault-tolerant perception pipeline combining a primary macro-object detector (YOLO11s), "
        "a targeted open-vocabulary zero-shot detector for personal micro-items (OWLv2 with nested-area IoU suppression), an Optical Character Recognition "
        "engine (EasyOCR), and an autoregressive caption generator (BLIP). Generated natural language descriptions are syntactically parsed via dependency chunking "
        "into discrete atomic claims. Each claim is mapped through a domain vocabulary ontology against empirical bounding boxes and text items, categorizing "
        "it into a deterministic five-tier epistemic taxonomy: supported, weak, not confirmed (hallucination candidate), unverifiable (out-of-ontology), or inference (action/attribute). "
        "A quantitative Evidence Consistency metric is dynamically computed to gauge factual grounding. Simultaneously, conversational interactions are governed "
        "by a deterministic, non-generative safety router with physical interlocks that prevent hazardous navigation or medical questions from reaching generative models.")
    add_p(doc,
        "Empirical evaluation on 400 held-out test images from Microsoft COCO val2017 demonstrates state-of-the-art reliability: supported claims achieve 98.16% precision "
        "against ground-truth human annotations, while 71.43% of unconfirmed claims are verified as true generative hallucinations. The complete end-to-end pipeline operates "
        "with a mean latency of 573.9 ms on CUDA hardware, with the claim verification engine adding only 5.1 ms of processing overhead (<1%). Furthermore, the system includes "
        "a privacy-preserving, dual-consent SQLite audit trail and an accessible Streamlit user interface featuring local Text-to-Speech.")

    p_kw = add_p(doc, "Keywords: ", bold=True)
    p_kw.runs[0].font.name = 'Arial'
    r_k = p_kw.add_run("Assistive AI, Computer Vision, Hallucination Auditing, Claim Verification, Multimodal NLP, Safety Gating, Explainable AI.")
    r_k.font.name = 'Times New Roman'
    r_k.italic = True

    doc.add_page_break()

    # ==========================================
    # TABLE OF CONTENTS & LISTS
    # ==========================================
    add_heading_1(doc, "Table of Contents")
    toc_items = [
        ("Abstract", "1"),
        ("List of Figures", "3"),
        ("List of Tables", "3"),
        ("List of Abbreviations", "3"),
        ("1 Introduction", "4"),
        ("  1.1 Proposed Model Context", "4"),
        ("  1.2 Research Objectives", "4"),
        ("  1.3 Research Gap", "5"),
        ("2 Literature Review", "5"),
        ("  2.1 Traditional Assistive Computer Vision Systems", "5"),
        ("  2.2 Vision-Language Models & Hallucination Challenges", "5"),
        ("  2.3 Multimodal Object Detection & Open-Vocabulary Detection", "5"),
        ("  2.4 Explainability, Verifiability & Claim Checking", "6"),
        ("  2.5 Safety Interlocks & Human-Centered Assistive Agents", "6"),
        ("3 Methodology", "7"),
        ("  3.1 Data Collection and Evaluation Dataset Split", "7"),
        ("  3.2 Image Acquisition and Signal Quality Preprocessing", "7"),
        ("  3.3 Multi-Branch Heterogeneous Feature Extraction", "8"),
        ("  3.4 2D Axis-Dominance Spatial Modeling", "9"),
        ("  3.5 Framework Architecture and Pipeline Flowchart", "9"),
        ("  3.6 Claim Extraction, Categorization and Verification Algorithms", "10"),
        ("  3.7 Question Routing and Deterministic Safety Interlocks", "11"),
        ("4 Results and Discussion", "12"),
        ("  4.1 Perception & Captioning Performance Analysis", "12"),
        ("  4.2 Claim Verification & Hallucination Auditing Analysis", "12"),
        ("  4.3 Subsystem Response Time and Latency Profile", "13"),
        ("  4.4 Environmental Robustness & Stress Analysis", "13"),
        ("  4.5 Qualitative Examples & Walkthroughs", "14"),
        ("  4.6 Comparative Analysis (AccessAI vs Conventional Models)", "14"),
        ("5 Graphical Analysis", "15"),
        ("6 System Architecture & User Interface Walkthrough", "17"),
        ("7 Individual Project Contributions", "18"),
        ("8 Conclusion and Future Scope", "18"),
        ("  8.1 Conclusion", "18"),
        ("  8.2 Limitations and Future Scope", "19"),
        ("References", "20")
    ]
    for title, pg in toc_items:
        p_toc = doc.add_paragraph()
        p_toc.paragraph_format.space_after = Pt(2)
        p_toc.paragraph_format.line_spacing = 1.15
        r_title = p_toc.add_run(title)
        r_title.font.name = 'Times New Roman'
        r_title.font.size = Pt(10.5)
        # Tab leader
        r_dots = p_toc.add_run(" " + "." * (80 - len(title) * 2) + " ")
        r_dots.font.color.rgb = RGBColor(148, 163, 184)
        r_pg = p_toc.add_run(pg)
        r_pg.font.name = 'Arial'
        r_pg.font.size = Pt(10)
        r_pg.bold = True

    doc.add_page_break()

    # ==========================================
    # LIST OF FIGURES, TABLES, ABBREVIATIONS
    # ==========================================
    add_heading_1(doc, "List of Figures")
    fig_items = [
        ("Figure 3.1: AccessAI COCO val2017 Dataset Split Distribution (500 Images)", "7"),
        ("Figure 3.2: AccessAI Sequential-Parallel Perception and Verification Architecture", "9"),
        ("Figure 4.1: Mean Processing Latency by Subsystem (400 Test Images on CUDA)", "13"),
        ("Figure 5.1: Performance Metrics Across Environmental Scene Conditions", "15"),
        ("Figure 5.2: Claim Verification Distribution Across 400 Test Captions (1,376 Total Claims)", "16"),
        ("Figure 5.3: Claim Verification Confusion Matrix (Checkable Claims)", "16"),
        ("Figure 5.4: Live Detection and Spatial Localization Snapshot with Bounding Boxes", "17")
    ]
    for caption, pg in fig_items:
        add_p(doc, f"{caption} .......................................................................... {pg}")

    add_heading_1(doc, "List of Tables")
    tbl_items = [
        ("Table 3.1: Dataset Sources, Image Conditioning, and Stratified Partitioning", "7"),
        ("Table 3.2: 5-Tier Claim Verification Taxonomy and Decision Criteria", "10"),
        ("Table 3.3: Deterministic Safety Routing Matrix and Execution Logic", "11"),
        ("Table 4.1: Overall Perception, Captioning, and Detection Benchmark on Test Split", "12"),
        ("Table 4.2: Claim Verification Performance Validated Against COCO Ground Truth", "12"),
        ("Table 4.3: Execution Latency Metrics Across Processing Subsystems (in ms)", "13"),
        ("Table 4.4: Stress Testing Metrics Across Diverse Environmental Conditions", "14"),
        ("Table 4.5: Conceptual and Functional Comparison: AccessAI vs State-of-the-Art Approaches", "14")
    ]
    for caption, pg in tbl_items:
        add_p(doc, f"{caption} .......................................................................... {pg}")

    add_heading_1(doc, "List of Abbreviations")
    abbrevs = [
        ("API", "Application Programming Interface"),
        ("BLEU", "Bilingual Evaluation Understudy"),
        ("CIDEr", "Consensus-based Image Description Evaluation"),
        ("COCO", "Common Objects in Context"),
        ("EXIF", "Exchangeable Image File Format"),
        ("IoU", "Intersection over Union"),
        ("mAP", "Mean Average Precision"),
        ("MLLM", "Multimodal Large Language Model"),
        ("NLP", "Natural Language Processing"),
        ("OCR", "Optical Character Recognition"),
        ("OWLv2", "Open-World Localization Version 2"),
        ("SQL", "Structured Query Language"),
        ("TTS", "Text-to-Speech"),
        ("VIP", "Visually Impaired Persons"),
        ("VLM", "Vision-Language Model"),
        ("VQA", "Visual Question Answering"),
        ("XAI", "Explainable Artificial Intelligence"),
        ("YOLO", "You Only Look Once")
    ]
    for abb, full in abbrevs:
        add_bullet(doc, f"• {abb}: ", full)

    doc.add_page_break()

    # ==========================================
    # SECTION 1: INTRODUCTION
    # ==========================================
    add_heading_1(doc, "1 Introduction")
    add_p(doc,
        "Visual impairment affects hundreds of millions of individuals worldwide, presenting daily challenges in navigation, "
        "object retrieval, reading environmental signage, and understanding unfamiliar physical spaces. While traditional mobility aids—such as "
        "the white cane and guide dogs—remain indispensable for immediate physical obstacle contact, they cannot convey semantic scene understanding, "
        "read printed textual labels, or answer descriptive questions about one's surroundings. The rapid convergence of computer vision, deep neural networks, "
        "and natural language processing has stimulated the development of assistive smartphone applications designed to serve as visual interpreters.")
    add_p(doc,
        "However, modern generative AI models exhibit a fundamental technological pathology: hallucination. Autoregressive language decoders "
        "frequently generate text that is syntactically fluent and contextually convincing, but factually false when compared against the visual input. "
        "In entertainment or generic conversational search, an occasional hallucination is a benign inconvenience. In assistive vision for visually impaired users, "
        "it is hazardous. A model that describes an open hallway when stairs are present, invents a pedestrian crossing, or misreads prescription dosage poses "
        "a direct threat to human safety. Furthermore, commercial assistive tools operate as opaque 'black boxes', offering no explanation for why a statement "
        "was made and providing no audit trail linking assertions back to verifiable visual features.")

    add_heading_2(doc, "1.1 Proposed Model Context")
    add_p(doc,
        "To establish a verifiable foundation for AI-assisted vision, the AccessAI project introduces an asynchronous, multi-branch architecture "
        "that enforces deterministic claim-to-evidence verification. AccessAI decouples the generative linguistic generation of BLIP from the empirical "
        "sensory observations of YOLO11s, OWLv2, and EasyOCR. Generated descriptions are programmatically deconstructed into atomic semantic claims, mapped to "
        "an ontological category dictionary, and checked against physical bounding boxes. Claims are classified into a transparent five-tier epistemic taxonomy: "
        "supported, weak, not confirmed, unverifiable, or inference. Furthermore, conversational queries regarding navigation, path clearance, and medication "
        "are intercepted by a deterministic rule-based safety router that refuses to hallucinate safe pathways, directing users to rely on their tactile aids.")

    add_heading_2(doc, "1.2 Research Objectives")
    add_bullet(doc, "Multi-Branch Perception: ", "To engineer an asynchronous, thread-safe computer vision pipeline that concurrently extracts macro-objects, micro-items, OCR text, and generative captions.")
    add_bullet(doc, "Deterministic Claim Verification: ", "To formulate an automated NLP decomposition algorithm that parses generative captions into discrete claims and verifies them against empirical bounding boxes.")
    add_bullet(doc, "Hallucination Auditing: ", "To establish a 5-tier epistemic status taxonomy and a quantitative Evidence Consistency Metric that reliably identifies generative hallucinations without requiring model introspection.")
    add_bullet(doc, "Safety Gating: ", "To design a deterministic question-routing interlock that guarantees physical safety and medical queries can never reach generative models.")
    add_bullet(doc, "Privacy & Accessibility: ", "To deploy an accessible Streamlit interface with offline Text-to-Speech (TTS) and a dual-consent, auto-purging SQLite audit trail.")

    add_heading_2(doc, "1.3 Research Gap")
    add_p(doc,
        "Prior art in assistive vision is partitioned into two distinct paradigms: (1) static, rule-based point tools (e.g., barcode scanners, OCR readers) "
        "that lack contextual visual description, and (2) monolithic end-to-end multimodal LLMs (e.g., GPT-4V, Gemini, Be My Eyes) that hallucinate freely and cannot "
        "provide evidence-based provenance. Academic benchmarks (e.g., POPE, CHAIR) quantify hallucination post-hoc on static datasets, but do not provide an inline, "
        "real-time runtime verifier. AccessAI bridges this critical gap by providing an explainable, lightweight verification layer that runs in 5.1 ms on-device, "
        "auditing every generated claim before presenting it to the user.")

    # ==========================================
    # SECTION 2: LITERATURE REVIEW
    # ==========================================
    add_heading_1(doc, "2 Literature Review")
    add_heading_2(doc, "2.1 Traditional Assistive Computer Vision Systems")
    add_p(doc,
        "Early assistive vision research focused primarily on isolated perceptual tasks. Mobile applications such as Microsoft Seeing AI [1] "
        "and Google Lookout [2] introduced modular channels for barcode recognition, document scanning, currency recognition, and short-text reading. "
        "While computationally efficient, these systems require users to manually select specialized modes and offer minimal holistic scene context. "
        "Furthermore, early scene description modes deployed lightweight convolutional neural networks that produced coarse, repetitive captions "
        "(e.g., 'a laptop on a table') without spatial relationships or text integration [3].")

    add_heading_2(doc, "2.2 Vision-Language Models & Hallucination Challenges")
    add_p(doc,
        "The advent of Vision-Language Models (VLMs) and transformer-based vision-language pretraining—such as BLIP [4], CLIP [5], and LLaVA [6]—revolutionized "
        "automated image description. By mapping visual tokens into joint linguistic embedding spaces, these models achieve unprecedented descriptive richness. "
        "However, extensive empirical studies [7], [8] demonstrate that modern VLMs suffer from pervasive object hallucination. Because autoregressive decoders "
        "maximize token probability sequences $P(w_t | w_{<t}, I)$, language priors often overpower ambiguous visual evidence. For example, in an image showing "
        "a dining room with a table, the model frequently hallucinates chairs, plates, or wine glasses that do not exist in the scene [9].")

    add_heading_2(doc, "2.3 Multimodal Object Detection & Open-Vocabulary Detection")
    add_p(doc,
        "Supervised object detectors like the YOLO series (YOLOv8 to YOLO11) [10] achieve exceptional precision and inference speeds on bounded label vocabularies "
        "(e.g., COCO's 80 classes). Nevertheless, visually impaired users regularly interact with personal utility micro-objects—such as pens, styluses, and earbuds—that "
        "fall outside standard benchmark classes. Recent advancements in open-vocabulary detection, notably OWLv2 [11], leverage contrastive image-text training to detect "
        "arbitrary objects guided by text queries. However, running open-vocabulary models over unbounded vocabularies remains computationally prohibitive on edge hardware, "
        "requiring strategic targeted query designs.")

    add_heading_2(doc, "2.4 Explainability, Verifiability & Claim Checking")
    add_p(doc,
        "Explainable AI (XAI) literature has increasingly sought methods to audit multimodal model predictions. Saliency maps and Grad-CAM [12] visualize attention "
        "weights, but offer zero semantic interpretability to a non-expert, let alone a blind user who cannot see visual heatmaps. Recent natural language claim-checking "
        "approaches—such as FactScore [13] and Woodpecker [14]—decompose text into facts and verify them using secondary LLM judges. However, using an LLM to check "
        "another LLM simply introduces second-order hallucinations while ballooning latency. A deterministic, rule-based claim verification engine grounded in observable "
        "bounding boxes represents a far more reliable engineering foundation.")

    add_heading_2(doc, "2.5 Safety Interlocks & Human-Centered Assistive Agents")
    add_p(doc,
        "Conversational visual agents that allow users to ask open-ended questions about their environment introduce severe safety vulnerabilities. In conversational "
        "benchmarks, generative models frequently display 'sycophancy' [15], affirming user premises even when hazardous (e.g., answering 'Yes, it looks safe to step down'). "
        "Human-computer interaction (HCI) research on accessibility [16] emphasizes that assistive tools must communicate system limitations transparently, distinguish between "
        "direct sensory observations and model inferences, and explicitly refuse life-critical navigation decisions.")

    doc.add_page_break()

    # ==========================================
    # SECTION 3: METHODOLOGY
    # ==========================================
    add_heading_1(doc, "3 Methodology")
    add_heading_2(doc, "3.1 Data Collection and Evaluation Dataset Split")
    add_p(doc,
        "To rigorously evaluate AccessAI without synthetic bias, the system was benchmarked against the standardized Microsoft COCO val2017 dataset [17]. "
        "A stratified sample of 500 images was partitioned using a fixed random seed into a development set (100 images) and a held-out test set (400 images). "
        "The dev split was utilized exclusively for tuning detection and OCR confidence thresholds, while the test split was reserved for final unseen evaluation. "
        "The 400 test images contain 2,738 ground-truth annotated objects across diverse environmental conditions (Table 3.1, Figure 3.1).")

    create_table(doc,
        ["Dataset Split", "Sample Count", "Ground Truth Objects", "Evaluation Purpose", "Condition Distribution"],
        [
            ["Development (Dev)", "100 images", "692 objects", "Threshold tuning & calibration", "Ordinary, cluttered, occluded"],
            ["Test (Held-Out)", "400 images", "2,738 objects", "Final unbiased system benchmarking", "240 Ordinary, 96 Cluttered, 117 Occluded"],
            ["Low-Light Test", "400 images", "2,738 objects", "Synthetic darkness stress testing (20% lum)", "100% Darkened stress split"]
        ],
        [1.5, 1.0, 1.3, 1.8, 1.6]
    )

    add_figure_image(doc, "figures/fig_3_1_dataset_split.png", "Figure 3.1: AccessAI COCO val2017 Dataset Split Distribution (500 Images)", width_inches=4.8)

    add_heading_2(doc, "3.2 Image Acquisition and Signal Quality Preprocessing")
    add_p(doc,
        "Raw image bytes captured from smartphone cameras or uploaded files undergo automated validation in accessai/preprocessing.py. "
        "Smartphone imagery frequently embeds rotational orientation within EXIF metadata rather than pixel matrices. AccessAI applies ImageOps.exif_transpose "
        "to ensure spatial coordinates align with true orientation. Images exceeding 1280 pixels are resized using Lanczos downsampling, while images smaller than "
        "32 pixels are rejected. Subsequently, two quantitative signal metrics are computed:")
    add_bullet(doc, "Photometric Luminance: ", "Computes the mean greyscale pixel intensity: mu_I = (1 / HW) * sum(I(x, y)). If mu_I < 50.0, a 'low_light' uncertainty flag is asserted.")
    add_bullet(doc, "Laplacian Focus Variance: ", "Evaluates image blur using the discrete 2D Laplacian operator: sigma^2_L = Var(del^2 I). If sigma^2_L < 60.0, a 'blurry' flag is asserted.")

    add_heading_2(doc, "3.3 Multi-Branch Heterogeneous Feature Extraction")
    add_p(doc,
        "To achieve low latency while decoupling observation from generation, AccessAI executes four parallel perception branches across a ThreadPoolExecutor (Figure 3.2):")
    add_bullet(doc, "Primary Macro-Object Detector (YOLO11s): ", "Scans the preprocessed RGB array for common objects across 80 COCO classes. Detections with confidence >= 0.25 are retained. Bounding boxes are classified into horizontal regions (left, center, right) based on centroid x-coordinates.")
    add_bullet(doc, "Targeted Micro-Item Detector (OWLv2): ", "Zero-shot transformer queried exclusively with a localized prompt set: Q = {'a pen', 'an earbud', 'an airpod'}. Candidate predictions are filtered using a nested Intersection-over-Smaller-Area ratio: O(A, B) = |A cap B| / min(|A|, |B|). If O(A, B) > 0.5, redundant nested boxes are eliminated.")
    add_bullet(doc, "Optical Character Recognition (EasyOCR): ", "Extracts alphanumeric text strings, polygon bounding boxes, and recognition confidence scores.")
    add_bullet(doc, "Generative Vision-Language Captioner (BLIP): ", "Generates an end-to-end natural language caption using 3-beam search conditional decoding, computing the average sequence log-probability score as caption confidence.")
    add_bullet(doc, "Fault Isolation: ", "Each module execution is wrapped in try-except handlers. If any individual module fails, the remaining modules complete execution, and the error is recorded in record.module_errors without halting the pipeline.")

    add_heading_2(doc, "3.4 2D Axis-Dominance Spatial Modeling")
    add_p(doc,
        "Because monocular smartphone cameras cannot reliably measure 3D metric distances, AccessAI models pairwise spatial relationships "
        "in the 2D image plane among the most confident detections (accessai/spatial.py). For bounding boxes a and b:")
    add_bullet(doc, "Overlap Check: ", "If IoU(a, b) >= 0.10, the relation is classified as 'overlaps'.")
    add_bullet(doc, "Centroid Vector: ", "Calculates centroid deltas: Delta_x = centroid_x(a) - centroid_x(b) and Delta_y = centroid_y(a) - centroid_y(b) (where y increases downward).")
    add_bullet(doc, "Dominant Direction: ", "If |Delta_x| >= |Delta_y|, the relation is 'left of' (Delta_x < 0) or 'right of' (Delta_x > 0). If |Delta_y| > |Delta_x|, the relation is 'above' (Delta_y < 0) or 'below' (Delta_y > 0). All descriptions are explicitly qualified: 'These are positions in the picture, not real distances.'")

    add_heading_2(doc, "3.5 Framework Architecture and Pipeline Flowchart")
    add_p(doc,
        "Figure 3.2 illustrates the complete end-to-end pipeline. The outputs of all perception branches are aggregated into an immutable AuditRecord dataclass "
        "prior to verification and response synthesis.")

    add_figure_image(doc, "figures/fig_3_2_pipeline_architecture.png", "Figure 3.2: AccessAI Sequential-Parallel Perception and Verification Architecture", width_inches=6.0)

    add_heading_2(doc, "3.6 Claim Extraction, Categorization and Verification Algorithms")
    add_p(doc,
        "The core innovation of AccessAI is the deterministic claim verification engine (accessai/claims.py). "
        "The generated caption is parsed using SpaCy dependency parsing (en_core_web_sm). Pronouns and photographic meta-nouns "
        "(e.g., 'photo', 'picture', 'scene', 'group') are purged. Compound noun phrases ending at chunk roots are resolved (e.g., 'stop sign', 'wine glass'). "
        "Numeric modifiers (dep_ == 'nummod') are parsed to extract stated counts. Each extracted claim is then verified against empirical detections and OCR text (Table 3.2).")

    create_table(doc,
        ["Epistemic Status", "Evaluation Logic / Trigger Condition", "Assistive Action / Spoken Qualification"],
        [
            ["Supported", "Confident detection (>=0.50) matching term ontology & count, or exact OCR match.", "Affirmed as firm empirical observation; linked to Evidence IDs."],
            ["Weak", "Detection confidence between 0.25 and 0.50, or stated count disagrees with detector.", "Flagged: 'Detected, but at low confidence' or count mismatch."],
            ["Not Confirmed", "Object class is within detector ontology, but 0 detections found.", "Flagged as potential hallucination: 'Mentioned, but could not be confirmed.'"],
            ["Unverifiable", "Object class is outside the detector's 80-class ontology & OCR text.", "Neutral handling: 'Outside detector categories; no evidence either way.'"],
            ["Inference", "Non-auxiliary verbs and descriptive adjectives (actions/attributes).", "Segregated as model interpretation: 'Not independently verified.'"]
        ],
        [1.3, 2.7, 2.5]
    )

    add_p(doc,
        "The system calculates an objective Evidence Consistency Score (C) representing the ratio of checkable claims that are supported:\n"
        "C = |Supported| / (|Supported| + |Weak| + |Not Confirmed|)\n"
        "Claims classified as unverifiable or inference are excluded from the denominator to avoid metric distortion.")

    add_heading_2(doc, "3.7 Question Routing and Deterministic Safety Interlocks")
    add_p(doc,
        "Conversational inquiries are processed by a rule-based query router (accessai/agent.py). "
        "To guarantee that hazardous questions can never be answered by hallucinating generative models, regex patterns enforce hard architectural interlocks (Table 3.3).")

    create_table(doc,
        ["Route Name", "Trigger Regex Patterns", "Handling Logic & System Response"],
        [
            ["Safety / Clear Path", "safe, hazard, obstacle, clear path, walk, cross", "Hard refusal: 'I can't tell you whether it is safe... Please rely on your cane or guide dog.' Injects detected objects list."],
            ["Medical / Dosage", "medicine, pill, dose, dosage, prescription", "Hard refusal: Refuses dosing interpretation. Reads visible text verbatim."],
            ["Distance / Depth", "how far, distance, close, meters, feet", "Refusal: 'I can't measure distance from a single photo.' Supplies 2D image position."],
            ["Count / Location", "how many, where is, presence, list objects", "Deterministic synthesis: Answers strictly using structured detection bounding boxes."],
            ["Text Reading", "read, what does it say, written, text, label", "Deterministic synthesis: Outputs confirmed OCR text items."],
            ["Generic Visual QA", "Any unrouted visual query (e.g., color, sky)", "Fallback to BLIP VQA: Prepend 'My best guess, which I can't verify...' and check answer with claim verifier."]
        ],
        [1.4, 2.3, 2.8]
    )

    doc.add_page_break()

    # ==========================================
    # SECTION 4: RESULTS AND DISCUSSION
    # ==========================================
    add_heading_1(doc, "4 Results and Discussion")
    add_heading_2(doc, "4.1 Perception & Captioning Performance Analysis")
    add_p(doc,
        "The complete AccessAI pipeline was evaluated on the 400 held-out test images on an NVIDIA CUDA platform. "
        "The primary YOLO11s detector achieved an mAP@.50 of 0.6557 and mAP@.50:.95 of 0.4930, with precision of 71.07% and recall of 60.37% "
        "at the conservative 0.25 confidence threshold. The BLIP caption generator achieved strong linguistic quality scores, recording a CIDEr score of 1.2021, "
        "BLEU-1 of 0.7033, BLEU-4 of 0.3443, and METEOR of 0.2667 (Table 4.1).")

    create_table(doc,
        ["Subsystem / Model", "Primary Metrics Evaluated", "Observed Value", "Standard Benchmark Reference"],
        [
            ["YOLO11s Object Detector", "mAP@.50 / mAP@.50:.95\nPrecision / Recall", "0.6557 / 0.4930\n71.07% / 60.37%", "COCO val2017 (2,738 GT objects)\nConfidence threshold = 0.25"],
            ["BLIP Caption Generator", "CIDEr / METEOR\nBLEU-1 / BLEU-4", "1.2021 / 0.2667\n0.7033 / 0.3443", "COCO human reference captions\nBeam search = 3"],
            ["EasyOCR Engine", "Character Sequence Similarity", "Clean: 99.44%\nLow-Contrast: 99.44%\nAngled (10 deg): 76.24%", "Rendered signage test benchmarks"]
        ],
        [1.8, 1.8, 1.6, 1.6]
    )

    add_heading_2(doc, "4.2 Claim Verification & Hallucination Auditing Analysis")
    add_p(doc,
        "Across 400 test images, the claim engine extracted 1,376 claims: 435 supported, 49 weak, 49 not confirmed, 431 unverifiable, and 412 inferences. "
        "Of the 400 images, 350 images contained checkable claims within the detector's ontology, yielding a mean per-image Evidence Consistency of 82.98%. "
        "To validate verification accuracy, claims were cross-referenced against ground-truth human annotations (Table 4.2).")

    create_table(doc,
        ["Claim Verification Category", "Extracted Count", "Ground Truth Present", "Ground Truth Absent", "Verification Precision / Accuracy"],
        [
            ["Supported Claims", "435 claims", "427 confirmed", "8 unconfirmed", "98.16% True Presence Precision"],
            ["Weak Claims (Low Conf/Count)", "49 claims", "44 confirmed", "5 absent", "89.80% True Presence Precision"],
            ["Not Confirmed (Hallucination)", "49 claims", "14 present (missed)", "35 confirmed absent", "71.43% True Hallucination Isolation"]
        ],
        [1.8, 1.0, 1.2, 1.2, 1.6]
    )

    add_p(doc,
        "Crucially, when AccessAI marks a claim as 'supported', the likelihood that the physical entity exists is 98.16%. "
        "Furthermore, 71.43% of claims marked as 'not confirmed' were verified to be non-existent in ground truth annotations, confirming the engine's capability "
        "to intercept hallucinations before they reach visually impaired users.")

    add_heading_2(doc, "4.3 Subsystem Response Time and Latency Profile")
    add_p(doc,
        "System latency is vital for real-time mobile assistive agents. AccessAI was profiled on CUDA hardware across all 400 test images (Table 4.3, Figure 4.1). "
        "The complete pipeline recorded a mean latency of 573.9 ms (median: 514.2 ms, p95: 931.6 ms). The deterministic claim verification engine required only "
        "5.1 ms on average, representing less than 1% of the overall computation budget.")

    create_table(doc,
        ["Processing Stage", "Mean Latency (ms)", "Median Latency (ms)", "P95 Latency (ms)", "% of Total Budget"],
        [
            ["Preprocessing & EXIF", "8.8 ms", "8.2 ms", "12.4 ms", "1.5%"],
            ["Object Detection (YOLO11s)", "247.1 ms", "237.2 ms", "350.5 ms", "43.1%"],
            ["Caption Generation (BLIP)", "565.0 ms", "505.6 ms", "923.7 ms", "Concurrent branch"],
            ["Optical Character Recognition", "169.4 ms", "134.1 ms", "399.6 ms", "Concurrent branch"],
            ["Claim Verification Engine", "5.1 ms", "4.0 ms", "9.0 ms", "0.9%"],
            ["Total End-to-End Pipeline", "573.9 ms", "514.2 ms", "931.6 ms", "100.0%"]
        ],
        [1.8, 1.2, 1.2, 1.2, 1.4]
    )

    add_figure_image(doc, "figures/fig_4_1_latency_breakdown.png", "Figure 4.1: Mean Processing Latency by Subsystem (400 Test Images on CUDA)", width_inches=5.2)

    add_heading_2(doc, "4.4 Environmental Robustness & Stress Analysis")
    add_p(doc,
        "To evaluate real-world resilience, images were partitioned into three natural environmental conditions—ordinary (240 images), "
        "cluttered (>=10 objects, 96 images), and occluded (IoU > 0.3, 117 images)—along with a synthetic low-light stress condition (20% luminance, 400 images). "
        "The automated quality assessor flagged 100% (400/400) of darkened images, asserting spoken warnings to alert users of reduced visual fidelity (Table 4.4, Figure 5.1).")

    create_table(doc,
        ["Environmental Condition", "Image Count", "Precision (%)", "Recall (%)", "Mean Evidence Consistency (%)"],
        [
            ["Ordinary Baseline", "240 images", "70.68%", "72.03%", "83.33%"],
            ["Cluttered (>=10 objects)", "96 images", "70.69%", "52.10%", "82.59%"],
            ["Occluded (Pairwise IoU > 0.3)", "117 images", "69.97%", "56.64%", "81.70%"],
            ["Low-Light Stress (20% lum)", "400 images", "74.10%", "55.08%", "81.73%"]
        ],
        [1.8, 1.1, 1.1, 1.1, 1.7]
    )

    add_heading_2(doc, "4.5 Explanation in Detail — Qualitative Examples and Walkthroughs")
    add_p(doc,
        "In a sample test scene (e.g., image 000000002157.jpg), the BLIP captioner generated: 'a table topped with wine glasses and plates of food'. "
        "The claim extraction module extracted three noun claims ('table', 'wine glasses', 'plates of food') and one action inference ('topped'). "
        "Cross-checking against YOLO11s detections confirmed 'dining table' (confidence 0.61, D7) and five 'wine glass' detections (best confidence 0.88, D1), "
        "marking both as supported. However, 'plates' could not be confirmed in the detector output, causing the system to assert: "
        "'The description mentions plates of food, but this could not be confirmed.' The spoken summary cleanly segregated the verified detections from the unconfirmed claim.")

    add_heading_2(doc, "4.6 Comparative Analysis (AccessAI vs Conventional Models)")
    add_p(doc,
        "Table 4.5 highlights the structural and operational differences between AccessAI and competing approaches.")

    create_table(doc,
        ["Capability / Feature", "Traditional Mobile Aids (Seeing AI)", "Monolithic Multimodal LLMs", "AccessAI Proposed Framework"],
        [
            ["Hallucination Auditing", "None; static black-box captions.", "None; stochastic hallucination.", "Deterministic 5-tier claim-to-evidence verifier."],
            ["Supported Claim Precision", "Unmeasured / low.", "Unverified.", "98.16% precision verified against ground truth."],
            ["Evidentiary Provenance", "None.", "None.", "Every claim linked to bounding box Evidence IDs."],
            ["Count Verification", "Frequently hallucinated.", "Prone to counting errors.", "Explicit comparison of stated numeral vs box count."],
            ["Safety Interlocks", "Static disclaimers in terms of use.", "Vulnerable to prompt injection.", "Deterministic regex router blocks generative path."],
            ["Privacy Protection", "Images uploaded to cloud servers.", "Cloud dependent.", "100% Local inference; dual-consent SQLite storage."]
        ],
        [1.6, 1.6, 1.6, 2.0]
    )

    doc.add_page_break()

    # ==========================================
    # SECTION 5: GRAPHICAL ANALYSIS
    # ==========================================
    add_heading_1(doc, "5 Graphical Analysis")
    add_p(doc,
        "Figures 5.1 through 5.4 present empirical evaluation visualizations generated from the held-out test benchmark: "
        "performance stability across environmental stress conditions, claim verification distributions, the confusion matrix for checkable claims, "
        "and an annotated detection snapshot.")

    add_figure_image(doc, "figures/fig_5_1_scene_conditions.png", "Figure 5.1: Performance Metrics Across Environmental Scene Conditions", width_inches=5.6)
    add_figure_image(doc, "figures/fig_5_2_claim_distribution.png", "Figure 5.2: Claim Verification Distribution Across 400 Test Captions (1,376 Total Claims)", width_inches=5.6)
    add_figure_image(doc, "figures/fig_5_3_confusion_matrix.png", "Figure 5.3: Claim Verification Confusion Matrix (Checkable Claims)", width_inches=4.8)
    add_figure_image(doc, "figures/snapshot_detection.jpg", "Figure 5.4: Live Detection and Spatial Localization Snapshot with Bounding Boxes (Image 000000002157.jpg)", width_inches=5.6)

    doc.add_page_break()

    # ==========================================
    # SECTION 6: SYSTEM ARCHITECTURE & UI WALKTHROUGH
    # ==========================================
    add_heading_1(doc, "6 System Architecture & User Interface Walkthrough")
    add_p(doc,
        "AccessAI is deployed as a high-contrast, accessible Streamlit dashboard (app.py) designed specifically for visually impaired users. "
        "The interface incorporates keyboard-navigable controls, large-text toggles, audible screen-reader aria labels, and native Text-to-Speech (TTS). "
        "The dashboard comprises six operational views:")
    add_bullet(doc, "Overview: ", "Provides high-level system readiness, camera controls, recent analysis metrics, and a pipeline activity banner.")
    add_bullet(doc, "Live Assist: ", "Direct camera capture and upload interface. Features automated speech output, high-contrast bounding box overlays, and expandable object tables.")
    add_bullet(doc, "Evidence & Audit: ", "Displays granular claim-to-evidence mappings, showing each extracted claim, its epistemic status badge, evidence bounding box IDs, and detailed explanations.")
    add_bullet(doc, "Model Agents: ", "Surfaces pipeline agent statuses and individual subsystem execution latencies for real-time diagnostics.")
    add_bullet(doc, "Image History: ", "Searchable audit database of past analyses, allowing users to inspect past evidence records or delete saved entries.")
    add_bullet(doc, "Settings: ", "Configures speech engines (offline pyttsx3 SAPI vs online gTTS), speech rates (100-280 wpm), high-contrast modes, and data retention policies.")

    add_heading_2(doc, "6.1 Privacy-Preserving Audit Trail with Dual-Consent Storage")
    add_p(doc,
        "Assistive devices regularly capture private documents, medical labels, and personal environments. AccessAI implements a decoupled, dual-consent storage architecture (accessai/audit.py). "
        "By default, zero data is written to disk. If the user opts into 'Save audit records', symbolic metadata (bounding boxes, text strings, consistency scores) is persisted in a local SQLite database (audit.sqlite3). "
        "Raw image bytes are saved only if the user explicitly grants a second, separate consent. In accordance with privacy best practices, all audit records older than 30 days are automatically purged upon store initialization.")

    # ==========================================
    # SECTION 7: INDIVIDUAL CONTRIBUTIONS
    # ==========================================
    add_heading_1(doc, "7 Individual Project Contributions")
    add_p(doc, "This project was designed, engineered, and evaluated with the following primary contributions:")
    add_bullet(doc, "Core Pipeline & Model Architecture: ", "Integration of YOLO11s, OWLv2 with nested-area IoU suppression, BLIP captioning, and EasyOCR within an asynchronous, thread-safe pipeline.")
    add_bullet(doc, "Claim Verification Engine: ", "Formulation and implementation of the SpaCy dependency parsing algorithm, vocabulary ontology mapping, 5-tier epistemic state machine, and consistency scoring metric.")
    add_bullet(doc, "Deterministic Safety Router: ", "Design of regex intent interlocks, non-generative safety and medical refusal handlers, and evidence-grounded conversational synthesis.")
    add_bullet(doc, "Empirical Evaluation & Benchmarking: ", "Execution of the 500-image COCO val2017 benchmark, latency profiling across subsystems, environmental condition stress testing, and claim validation.")
    add_bullet(doc, "UI & Accessibility Engineering: ", "Development of the Streamlit dashboard, offline/online TTS subprocess architecture, high-contrast themes, and dual-consent SQLite persistence.")

    # ==========================================
    # SECTION 8: CONCLUSION AND FUTURE SCOPE
    # ==========================================
    add_heading_1(doc, "8 Conclusion and Future Scope")
    add_heading_2(doc, "8.1 Conclusion")
    add_p(doc,
        "The AccessAI project demonstrates that practical, explainable, and safe visual assistance can be engineered by combining "
        "parallel multi-modal perception with deterministic claim verification and safety gating. Rather than trusting unconstrained generative language models, "
        "AccessAI audits every assertion against observable sensory evidence, achieving 98.16% precision on supported claims while isolating 71.43% of hallucinations. "
        "Operating with a mean latency of 573.9 ms on CUDA hardware with under 5.1 ms verification overhead, the system proves that trustworthy, real-time visual "
        "assistance is achievable without cloud dependence, protecting visually impaired users from hazardous misguidance.")

    add_heading_2(doc, "8.2 Limitations and Future Scope")
    add_p(doc,
        "While highly effective, current limitations offer valuable directions for future enhancement: (1) The primary object detector is bounded by the 80 COCO classes, "
        "requiring expanded open-vocabulary cascades for broader utility objects; (2) Spatial modeling is confined to 2D image coordinates, which could be augmented by "
        "monocular depth estimation networks (e.g., Depth Anything) to convey physical proximity; (3) Conversational input currently relies on typed text, which will be "
        "expanded to include offline speech recognition (e.g., Whisper); and (4) User testing with visually impaired individuals should be conducted following institutional ethical review.")

    doc.add_page_break()

    # ==========================================
    # REFERENCES (IEEE Style)
    # ==========================================
    add_heading_1(doc, "References")
    references = [
        "[1] Microsoft, 'Seeing AI: Talking camera for the blind,' Microsoft AI Research, 2017. [Online]. Available: https://www.microsoft.com/en-us/ai/seeing-ai",
        "[2] Google, 'Lookout: Assisted vision with computer vision,' Google Accessibility, 2019. [Online]. Available: https://blog.google/outreach-initiatives/accessibility/lookout-app/",
        "[3] D. Gurari et al., 'VizWiz grand challenge: Answering visual questions from blind people,' in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2018, pp. 3608–3617.",
        "[4] J. Li, D. Li, S. Xiong, and S. Hoi, 'BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation,' in Proc. 39th Int. Conf. Machine Learning (ICML), 2022, pp. 12888–12900.",
        "[5] A. Radford et al., 'Learning transferable visual models from natural language supervision,' in Proc. 38th Int. Conf. Machine Learning (ICML), 2021, pp. 8748–8763.",
        "[6] H. Liu, C. Li, Q. Wu, and Y. J. Lee, 'Visual instruction tuning,' Advances in Neural Information Processing Systems (NeurIPS), vol. 36, pp. 34892–34916, 2023.",
        "[7] Y. Li et al., 'Evaluating object hallucination in large vision-language models,' in Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP), 2023, pp. 292–305.",
        "[8] F. Liu et al., 'Aligning large multimodal models with factually augmented RLHF,' arXiv preprint arXiv:2309.14525, 2023.",
        "[9] A. Rohrbach, L. A. Hendricks, K. Burns, T. Darrell, and K. Saenko, 'Object hallucination in image captioning,' in Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP), 2018, pp. 4035–4045.",
        "[10] G. Jocher et al., 'Ultralytics YOLO11,' Ultralytics Inc., 2024. [Online]. Available: https://github.com/ultralytics/ultralytics",
        "[11] M. Minderer, A. Gritsenko, and N. Houlsby, 'Scaling open-vocabulary object detection,' Advances in Neural Information Processing Systems (NeurIPS), vol. 36, pp. 72983–73014, 2023.",
        "[12] R. R. Selvaraju et al., 'Grad-CAM: Visual explanations from deep networks via gradient-based localization,' in Proc. IEEE Int. Conf. Computer Vision (ICCV), 2017, pp. 618–626.",
        "[13] S. Min et al., 'FActScore: Fine-grained atomic evaluation of factual precision in long-form text generation,' in Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP), 2023.",
        "[14] S. Yin et al., 'Woodpecker: Hallucination correction for multimodal large language models,' arXiv preprint arXiv:2310.16045, 2023.",
        "[15] M. Sharma et al., 'Towards understanding sycophancy in large language models,' arXiv preprint arXiv:2310.13548, 2023.",
        "[16] T. Bigham et al., 'VizWiz: Nearly real-time answers to visual questions from blind users,' in Proc. ACM Symp. User Interface Software and Technology (UIST), 2010, pp. 333–342.",
        "[17] T.-Y. Lin et al., 'Microsoft COCO: Common objects in context,' in Proc. European Conf. Computer Vision (ECCV), Springer, 2014, pp. 740–755.",
        "[18] X. Wu et al., 'A survey of human-in-the-loop for machine learning,' arXiv preprint arXiv:2108.00941, 2022."
    ]
    for ref in references:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.left_indent = Inches(0.4)
        p_ref.paragraph_format.first_line_indent = Inches(-0.4)
        p_ref.paragraph_format.space_after = Pt(4)
        p_ref.paragraph_format.line_spacing = 1.15
        r = p_ref.add_run(ref)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(9.5)

    output_path = Path("AccessAI_Project_Report.docx")
    doc.save(output_path)
    print(f"Report generated successfully: {output_path}")

if __name__ == "__main__":
    generate_report()
