"""AccessAI Streamlit dashboard.

The UI is intentionally thin: all analysis, evidence, persistence, and speech
are delegated to the existing AccessAI services.
"""

from __future__ import annotations

import hashlib
import html
import json

import pandas as pd
import streamlit as st

from accessai.agent import Agent
from accessai.audit import AuditStore
from accessai.config import DEFAULT_CONFIG
from accessai.pipeline import AccessAIPipeline
from accessai.preprocessing import InvalidImageError, draw_boxes
from accessai.tts import SpeechError, synthesize_offline, synthesize_online

STATUS_LABELS = {
    "supported": "Supported",
    "weak": "Weak",
    "not_confirmed": "Not confirmed",
    "unverifiable": "Cannot be checked",
    "inference": "AI interpretation",
}

FLAG_LABELS = {
    "low_light": "The image is dark",
    "blurry": "The image looks blurry",
    "low_resolution": "The image is low resolution",
    "no_detections": "No objects detected",
    "low_confidence_detections": "Some detections are low confidence",
    "low_confidence_text": "Some text may be misread",
    "low_confidence_caption": "The caption model was unsure",
    "unconfirmed_claims": "The caption contains unconfirmed claims",
    "module_failure": "A processing module failed",
}

st.set_page_config(page_title="AccessAI", page_icon="◉", layout="wide")


@st.cache_resource(show_spinner=False)
def load_pipeline() -> AccessAIPipeline:
    return AccessAIPipeline(DEFAULT_CONFIG)


@st.cache_resource(show_spinner=False)
def load_store() -> AuditStore:
    return AuditStore(DEFAULT_CONFIG)


def inject_styles(dark: bool, large_text: bool) -> None:
    theme = "dark" if dark else "light"
    st.markdown(
        f"""
        <style>
        :root {{
          --bg: {"#090D20" if dark else "#F8F6FF"};
          --surface: {"#131A35" if dark else "#FFFFFF"};
          --surface-soft: {"#1D2548" if dark else "#F1ECFF"};
          --text: {"#F5F3FF" if dark else "#11143F"};
          --muted: {"#B8B8D7" if dark else "#5C5C7A"};
          --border: {"#34345F" if dark else "#DDD6F7"};
          --primary: {"#A78BFA" if dark else "#7135E8"};
          --primary-soft: {"#38256C" if dark else "#E8DEFF"};
          --success: #22C55E;
          --shadow: {"0 12px 30px rgba(0,0,0,.28)" if dark else "0 10px 24px rgba(78,47,150,.08)"};
        }}
        .stApp {{ background: var(--bg); color: var(--text); }}
        [data-testid="stHeader"] {{ background: transparent; }}
        section[data-testid="stSidebar"] {{ min-width: 250px; max-width: 250px; }}
        [data-testid="stSidebar"] {{ background: {"#0C1025" if dark else "#F1ECFF"}; border-right: 1px solid var(--border); }}
        [data-testid="stSidebar"] * {{ color: var(--text); }}
        [data-testid="stSidebar"] > div:first-child {{ padding: 1.15rem .8rem; }}
        [data-testid="stSidebar"] [data-testid="stRadioGroup"] {{
          gap: .3rem; margin-top: .2rem;
        }}
        [data-testid="stSidebar"] [data-testid="stRadioGroup"] > div > label {{
          align-items: center; border-radius: 9px; display: flex; font-size: .9rem;
          font-weight: 600; min-height: 2.45rem; padding: .25rem .7rem;
          transition: background .15s ease, color .15s ease;
        }}
        [data-testid="stSidebar"] [data-testid="stRadioGroup"] > div > label:hover {{
          background: {"#241D4A" if dark else "#E9DEFF"};
        }}
        [data-testid="stSidebar"] [data-testid="stRadioGroup"] > div > label:has(input:checked) {{
          background: {"#4A327D" if dark else "#DFCEFF"};
          color: {"#FFFFFF" if dark else "#10143A"}; font-weight: 750;
        }}
        [data-testid="stSidebar"] [data-testid="stRadioGroup"] > div > label > div > div:first-child {{
          display: none;
        }}
        [data-testid="stSidebar"] [data-testid="stRadioGroup"] > div > label > div {{
          width: 100%;
        }}
        [data-testid="stSidebar"] [data-testid="stRadioGroup"] > div > label p {{
          align-items: center; display: flex; gap: .75rem; margin: 0;
        }}
        [data-testid="stSidebar"] [data-testid="stRadioGroup"] > div > label::before {{
          background-position: center; background-repeat: no-repeat; background-size: 19px 19px;
          content: ""; flex: 0 0 19px; height: 19px; margin-right: .05rem; width: 19px;
        }}
        [data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(1) > label::before {{
          background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%236C2BD9' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z'/%3E%3Cpolyline points='9 22 9 12 15 12 15 22'/%3E%3C/svg%3E");
        }}
        [data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(2) > label::before {{
          background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%236C2BD9' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3z'/%3E%3Ccircle cx='12' cy='13' r='3'/%3E%3C/svg%3E");
        }}
        [data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(3) > label::before {{
          background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%236C2BD9' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect width='18' height='18' x='3' y='3' rx='2'/%3E%3Ccircle cx='9' cy='9' r='2'/%3E%3Cpath d='m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21'/%3E%3C/svg%3E");
        }}
        [data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(4) > label::before {{
          background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%236C2BD9' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M20 13c0 5-3.5 7.5-8 9-4.5-1.5-8-4-8-9V5l8-3 8 3z'/%3E%3Cpath d='m9 12 2 2 4-4'/%3E%3C/svg%3E");
        }}
        [data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(5) > label::before {{
          background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%236C2BD9' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='5' r='3'/%3E%3Ccircle cx='5' cy='19' r='3'/%3E%3Ccircle cx='19' cy='19' r='3'/%3E%3Cpath d='M12 8v4m0 0-7 4m7-4 7 4'/%3E%3C/svg%3E");
        }}
        [data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(6) > label::before {{
          background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%236C2BD9' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='3'/%3E%3Cpath d='M19.4 15a1.7 1.7 0 0 0 .34 1.88l.06.06-1.7 1.7-.06-.06a1.7 1.7 0 0 0-1.88-.34 1.7 1.7 0 0 0-1.03 1.56V22h-2.4v-.09a1.7 1.7 0 0 0-1.03-1.56 1.7 1.7 0 0 0-1.88.34l-.06.06-1.7-1.7.06-.06A1.7 1.7 0 0 0 8.4 15a1.7 1.7 0 0 0-1.56-1.03H6v-2.4h.84A1.7 1.7 0 0 0 8.4 10a1.7 1.7 0 0 0-.34-1.88L8 8.06l1.7-1.7.06.06a1.7 1.7 0 0 0 1.88.34A1.7 1.7 0 0 0 12.67 5.2V4h2.4v1.2a1.7 1.7 0 0 0 1.03 1.56 1.7 1.7 0 0 0 1.88-.34l.06-.06 1.7 1.7-.06.06A1.7 1.7 0 0 0 19.4 10a1.7 1.7 0 0 0 1.56 1.03H22v2.4h-1.04A1.7 1.7 0 0 0 19.4 15z'/%3E%3C/svg%3E");
        }}
        .sidebar-footer {{
          color: var(--muted); font-size: .78rem; line-height: 1.25; margin-top: 6rem;
          text-align: center;
        }}
        .sidebar-footer svg {{ display: block; height: 104px; margin: 0 auto .35rem; width: 100%; }}
        h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"] {{ color: var(--text); }}
        .block-container {{ padding: 2rem 3rem 3rem; max-width: 1500px; }}
        .brand {{ font-size: 1.7rem; font-weight: 800; letter-spacing: -.04em; }}
        .brand span {{ color: var(--primary); }}
        .subtitle {{ color: var(--muted); font-size: .82rem; }}
        .page-title {{ font-size: 2rem; font-weight: 800; margin: 0; }}
        .page-description {{ color: var(--muted); margin-top: .2rem; }}
        .card, .metric-card {{
          background: var(--surface); border: 1px solid var(--border); border-radius: 18px;
          padding: 1.1rem; box-shadow: var(--shadow); margin-bottom: 1rem;
        }}
        .metric-card {{ min-height: 112px; }}
        .metric-label {{ color: var(--muted); font-size: .84rem; }}
        .metric-value {{ color: var(--text); font-size: 1.85rem; font-weight: 800; margin-top: .25rem; }}
        .metric-icon {{ color: var(--primary); font-size: 1.5rem; float: right; }}
        .section-title {{ font-size: 1.15rem; font-weight: 750; margin-bottom: .65rem; }}
        .pill {{ display:inline-block; padding:.25rem .6rem; border-radius:999px; background:var(--primary-soft); color:var(--primary); font-size:.75rem; font-weight:700; }}
        .success {{ color: var(--success); font-weight: 700; }}
        .muted {{ color: var(--muted); }}
        .pipeline {{ display:flex; flex-wrap:wrap; gap:.5rem; align-items:center; }}
        .pipeline-step {{ background:var(--surface-soft); border:1px solid var(--border); border-radius:12px; padding:.65rem .75rem; font-size:.78rem; font-weight:650; }}
        .pipeline-arrow {{ color:var(--primary); font-weight:800; }}
        div[data-testid="stMetric"] {{ background: var(--surface); border:1px solid var(--border); border-radius:16px; padding:1rem; }}
        div[data-testid="stMetric"] label {{ color:var(--muted); }}
        .stButton > button, .stDownloadButton > button {{ border-radius:10px; border:1px solid var(--border); color:var(--text); background:var(--surface); }}
        .stButton > button[kind="primary"] {{ background:linear-gradient(135deg,#7135E8,#9B5CF6); color:#fff; border:0; }}
        .stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stTextArea textarea {{ background:var(--surface); color:var(--text); border-color:var(--border); }}
        .small-note {{ color:var(--muted); font-size:.78rem; }}
        @media (max-width: 800px) {{ .block-container {{ padding:1rem; }} .page-title {{ font-size:1.55rem; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    if large_text:
        st.markdown("<style>html { font-size: 112%; }</style>", unsafe_allow_html=True)
    st.session_state["theme"] = theme


def settings_state() -> dict:
    defaults = {
        "engine": "offline", "rate": 170, "autoplay": True, "save_audit": False,
        "save_image": False, "large_text": False, "high_contrast": False,
        "volume": 80, "repeat": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    return {key: st.session_state[key] for key in defaults}


def speak(text: str, engine: str, rate: int) -> tuple[bytes, str] | None:
    try:
        return synthesize_online(text) if engine == "online" else synthesize_offline(text, rate)
    except SpeechError as exc:
        st.error(f"Speech is unavailable, but the text above is complete. {exc}")
        return None


def persist(record, image_bytes: bytes | None, settings: dict) -> None:
    if settings["save_audit"]:
        load_store().save(record, image_bytes if settings["save_image"] else None)


def render_audio(key: str, audio, settings: dict) -> None:
    if not audio:
        return
    autoplay = settings["autoplay"] and st.session_state.get("fresh_audio") == key
    st.audio(audio[0], format=audio[1], autoplay=autoplay)
    if autoplay:
        st.session_state.fresh_audio = None


def get_image_bytes() -> bytes | None:
    source = st.radio("Input source", ["Upload a photo", "Use the camera"], horizontal=True)
    if source == "Upload a photo":
        image = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png", "webp", "bmp"])
    else:
        image = st.camera_input("Take a photo")
    return image.getvalue() if image else None


def run_analysis(data: bytes, settings: dict) -> None:
    digest = hashlib.sha256(data).hexdigest()
    if st.session_state.get("digest") == digest:
        return
    for key in ("prepared", "record", "audio", "qa_audio"):
        st.session_state.pop(key, None)
    st.session_state.digest = digest
    with st.spinner("Analysing the image. The first run also loads the AI models."):
        try:
            prepared, record = load_pipeline().analyze_bytes(data)
        except InvalidImageError as exc:
            st.session_state.input_error = str(exc)
            return
    st.session_state.pop("input_error", None)
    st.session_state.prepared, st.session_state.record = prepared, record
    st.session_state.audio = speak(record.response, settings["engine"], settings["rate"])
    st.session_state.fresh_audio = "description"
    persist(record, data, settings)


def metric(label: str, value: str, icon: str) -> None:
    st.markdown(
        f'<div class="metric-card"><span class="metric-icon">{icon}</span>'
        f'<div class="metric-label">{html.escape(label)}</div><div class="metric-value">{html.escape(value)}</div></div>',
        unsafe_allow_html=True,
    )


def record_metrics(record) -> None:
    if not record:
        return
    cols = st.columns(4)
    values = [
        ("Detected objects", str(len(record.detections)), "◈"),
        ("OCR regions", str(len(record.text_items)), "T"),
        ("Evidence consistency", f"{record.consistency:.0%}" if record.consistency is not None else "—", "✓"),
        ("Processing time", f"{record.timings_ms.get('total', 0):.0f} ms", "ϟ"),
    ]
    for col, (label, value, icon) in zip(cols, values):
        with col:
            metric(label, value, icon)


def render_pipeline(record=None) -> None:
    st.markdown('<div class="section-title">AI agent pipeline</div>', unsafe_allow_html=True)
    names = ["Image input", "Preprocessing", "Object detection", "Image captioning", "OCR", "Spatial analysis", "Evidence"]
    st.markdown(
        '<div class="pipeline">' + "".join(
            f'<span class="pipeline-step">{name}</span>' + ('<span class="pipeline-arrow">→</span>' if i < len(names)-1 else "")
            for i, name in enumerate(names)
        ) + "</div>",
        unsafe_allow_html=True,
    )
    if record:
        st.caption(" · ".join(f"{name}: {record.timings_ms.get(key, 0):.0f} ms" for name, key in [
            ("Detection", "detection"), ("Caption", "caption"), ("OCR", "ocr"), ("Claims", "claim_check")
        ] if key in record.timings_ms))


def render_evidence(record, prepared=None) -> None:
    if not record:
        st.info("Analyse an image to populate evidence and audit details.")
        return
    if record.uncertainty_flags:
        st.warning("Uncertainty: " + "; ".join(FLAG_LABELS.get(flag, flag) for flag in record.uncertainty_flags))
    for module, error in record.module_errors.items():
        st.error(f"The {module} module failed: {error}")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown('<div class="card"><div class="section-title">Scene description</div>', unsafe_allow_html=True)
        st.write(record.response)
        if record.caption:
            st.caption(f"Caption confidence: {record.caption.confidence if record.caption.confidence is not None else 'not reported'}")
        st.markdown("</div>", unsafe_allow_html=True)
        if prepared is not None:
            st.image(draw_boxes(prepared.rgb, record.detections, record.text_items), use_container_width=True)
    with right:
        st.markdown('<div class="card"><div class="section-title">Detected objects</div>', unsafe_allow_html=True)
        if record.detections:
            st.dataframe(pd.DataFrame([{"Object": d.label, "Confidence": d.confidence, "Region": d.region, "Box": d.bbox} for d in record.detections]), hide_index=True, use_container_width=True)
        else:
            st.write("No objects were detected.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="section-title">Extracted text</div>', unsafe_allow_html=True)
        st.write(", ".join(t.text for t in record.text_items) or "No readable text was found.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="section-title">Claim-to-evidence mapping</div>', unsafe_allow_html=True)
    if record.claims:
        st.dataframe(pd.DataFrame([{"Claim": c.text, "Result": STATUS_LABELS.get(c.status, c.status), "Evidence": ", ".join(c.evidence_ids) or "None", "Explanation": c.explanation} for c in record.claims]), hide_index=True, use_container_width=True)
    else:
        st.write("No claims could be checked.")
    st.markdown("</div>", unsafe_allow_html=True)
    with st.expander("Spatial relationships and model details"):
        for relation in record.relations:
            st.write(relation.description)
        st.json({"device": record.device, "models": record.models, "timings_ms": record.timings_ms})
    st.download_button("Download audit report", json.dumps(record.to_dict(), indent=2), f"accessai_audit_{record.record_id}.json", "application/json")


def page_overview(settings: dict) -> None:
    record = st.session_state.get("record")
    records = load_store().list_records()
    st.markdown('<div class="page-title">Overview</div><div class="page-description">Assistive visual intelligence with transparent evidence.</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    summary = [("Analysed images", str(len(records) + (1 if record and not settings["save_audit"] else 0)), "▧"),
               ("Detected objects", str(len(record.detections)) if record else "—", "◈"),
               ("OCR text regions", str(len(record.text_items)) if record else "—", "T"),
               ("Evidence consistency", f"{record.consistency:.0%}" if record and record.consistency is not None else "—", "✓")]
    for col, item in zip(cols, summary):
        with col: metric(*item)
    st.markdown('<div class="card"><div class="section-title">Recent analysis</div>', unsafe_allow_html=True)
    if record:
        st.write(record.response)
        render_audio("description", st.session_state.get("audio"), settings)
        record_metrics(record)
    else:
        st.info("Upload or capture an image from Live Assist to begin.")
    st.markdown("</div>", unsafe_allow_html=True)
    render_pipeline(record)
    st.markdown('<div class="card"><div class="section-title">System status</div><span class="success">● Ready</span>  Local pipeline available; models load when first used.</div>', unsafe_allow_html=True)


def page_live_assist(settings: dict) -> None:
    st.markdown('<div class="page-title">Live Assist</div><div class="page-description">Capture an image and receive an explainable description.</div>', unsafe_allow_html=True)
    data = get_image_bytes()
    if data:
        run_analysis(data, settings)
    if st.session_state.get("input_error"):
        st.error(st.session_state.input_error)
        return
    record, prepared = st.session_state.get("record"), st.session_state.get("prepared")
    if not record:
        st.info("Choose an image source above. Live streaming is not implemented by the current backend; camera capture is supported.")
        return
    a, b = st.columns([1.15, 1])
    with a:
        if prepared: st.image(prepared.rgb, use_container_width=True)
        st.button("Start camera", disabled=True, help="Use the camera capture above.")
        st.button("Capture image", type="primary")
    with b:
        st.markdown('<div class="card"><div class="section-title">Live scene description</div>', unsafe_allow_html=True)
        st.write(record.response)
        render_audio("description", st.session_state.get("audio"), settings)
        st.button("Speak again", on_click=lambda: st.session_state.update(fresh_audio="description"))
        st.markdown("</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Detected objects**")
        st.dataframe(pd.DataFrame([{"Object": d.label, "Confidence": d.confidence} for d in record.detections]), hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**Extracted text**")
        st.write(", ".join(t.text for t in record.text_items) or "None")
    with c3:
        st.markdown("**Spatial relationships**")
        for relation in record.relations: st.write(relation.description)
    render_pipeline(record)


def page_history() -> None:
    st.markdown('<div class="page-title">Image History</div><div class="page-description">Saved analyses and their evidence reports.</div>', unsafe_allow_html=True)
    records = load_store().list_records()
    cols = st.columns(4)
    for col, item in zip(cols, [("Total analyses", str(len(records)), "▧"), ("Objects detected", "Available per report", "◈"), ("Images with OCR", "Available per report", "T"), ("Average confidence", "Available per report", "◎")]):
        with col: metric(*item)
    query = st.text_input("Search by scene, object, or OCR text")
    if not records:
        st.info("No audit records are saved. Enable saving in Settings before analysing an image.")
        return
    filtered = [row for row in records if not query or query.lower() in row["response"].lower()]
    st.dataframe(pd.DataFrame(filtered)[["record_id", "timestamp", "consistency", "response"]], hide_index=True, use_container_width=True)
    if filtered:
        choice = st.selectbox("Open a saved analysis", [row["record_id"] for row in filtered])
        selected = load_store().get(choice)
        if selected:
            st.json(selected, expanded=False)
    if st.button("Delete all saved records"):
        load_store().delete_all()
        st.rerun()


def page_evidence(settings: dict) -> None:
    st.markdown('<div class="page-title">Evidence & Audit</div><div class="page-description">Verify each generated claim against observable model output.</div>', unsafe_allow_html=True)
    record = st.session_state.get("record")
    record_metrics(record)
    render_evidence(record, st.session_state.get("prepared"))
    if record:
        audio = st.session_state.get("audio")
        if st.button("Listen to description"):
            st.session_state.fresh_audio = "description"
        render_audio("description", audio, settings)


def page_agents() -> None:
    st.markdown('<div class="page-title">Model Agents</div><div class="page-description">Transparent view of the local pipeline components.</div>', unsafe_allow_html=True)
    record = st.session_state.get("record")
    agents = [("Orchestrator", "Coordinates the analysis pipeline"), ("Preprocessing", "Normalises and validates images"), ("YOLO object detection", "Detects supported objects"), ("BLIP captioning", "Generates a scene caption"), ("OCR", "Extracts visible text"), ("Spatial analysis", "Derives 2D image relationships"), ("Evidence aggregation", "Checks claims against outputs"), ("Response generation", "Builds the accessible response"), ("Text-to-speech", "Speaks the response when requested")]
    for row in range(0, len(agents), 3):
        cols = st.columns(3)
        for col, (name, description) in zip(cols, agents[row:row+3]):
            with col:
                status = "Processed" if record else "Ready"
                st.markdown(f'<div class="card"><div class="section-title">{name}</div><div class="muted">{description}</div><p class="success">● {status}</p></div>', unsafe_allow_html=True)
    render_pipeline(record)
    if record:
        st.dataframe(pd.DataFrame([{"Agent": name, "Time": f"{record.timings_ms.get(key, 0):.1f} ms"} for name, key in [("Object detection", "detection"), ("Captioning", "caption"), ("OCR", "ocr"), ("Claim checking", "claim_check")]]), hide_index=True, use_container_width=True)
    else:
        st.info("Processing metrics appear after an analysis. Hardware metrics are not exposed by the current backend.")


def page_settings() -> None:
    st.markdown('<div class="page-title">Settings</div><div class="page-description">Customize accessibility, voice, privacy, and appearance.</div>', unsafe_allow_html=True)
    st.session_state.engine = "online" if st.radio("Voice engine", ["Offline (private)", "Online (gTTS)"], index=1 if st.session_state.engine == "online" else 0, horizontal=True).startswith("Online") else "offline"
    st.session_state.rate = st.slider("Speech rate", 100, 280, st.session_state.rate, step=10)
    st.session_state.autoplay = st.toggle("Speak results automatically", st.session_state.autoplay)
    st.toggle("Large text", key="large_text")
    st.toggle("High contrast mode", key="high_contrast")
    st.toggle("Save audit records on this computer", key="save_audit")
    st.toggle("Also save images", key="save_image", disabled=not st.session_state.save_audit)
    st.session_state.volume = st.slider("Audio volume", 0, 100, st.session_state.volume)
    st.caption(f"Saved records are purged after {DEFAULT_CONFIG.retention_days} days. Images require separate consent.")
    if st.button("Reset to defaults"):
        for key in ("engine", "rate", "autoplay", "save_audit", "save_image", "large_text", "high_contrast", "volume", "repeat"):
            st.session_state.pop(key, None)
        st.rerun()


def render_sidebar() -> tuple[str, dict]:
    settings = settings_state()
    with st.sidebar:
        st.markdown('<div class="brand">Access<span>AI</span></div><div class="subtitle">Assistive Visual Intelligence</div>', unsafe_allow_html=True)
        st.divider()
        pages = ["Overview", "Live Assist", "Image History", "Evidence & Audit", "Model Agents", "Settings"]
        current = st.session_state.get("page", "Overview")
        selected = st.radio("Navigation", pages, index=pages.index(current), label_visibility="collapsed")
        st.session_state.page = selected
        st.divider()
        st.toggle("Dark mode", key="dark_mode")
        st.toggle("Large text", key="sidebar_large_text")
        st.session_state.large_text = st.session_state.sidebar_large_text
        st.markdown(
            '<div class="sidebar-footer">'
            '<svg viewBox="0 0 240 120" aria-hidden="true">'
            '<path d="M0 74 C36 88 47 108 88 96 C127 84 139 61 174 60 C202 59 220 71 240 78 V120 H0Z" fill="#DFCEFF"/>'
            '<path d="M0 85 C36 99 50 112 89 101 C127 90 143 68 177 68 C204 68 220 79 240 85 V120 H0Z" fill="#E9DEFF"/>'
            '<g fill="none" stroke="#6C2BD9" stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5">'
            '<circle cx="103" cy="48" r="4"/><path d="M102 54c-5 8-7 16-6 25l-2 17m8-41 8 12 10 1m-16-12-8 8-7 15m15-3 8 17m-8-17-10 18m23-28 15 23"/></g>'
            '</svg>'
            'AI-Powered Visual<br>Assistance<br>for Everyday Life</div>',
            unsafe_allow_html=True,
        )
        st.caption("AI can be wrong. It cannot determine whether a route is safe.")
    return selected, settings


def main() -> None:
    page, settings = render_sidebar()
    inject_styles(st.session_state.get("dark_mode", False), st.session_state.get("large_text", False))
    if page == "Overview": page_overview(settings)
    elif page == "Live Assist": page_live_assist(settings)
    elif page == "Image History": page_history()
    elif page == "Evidence & Audit": page_evidence(settings)
    elif page == "Model Agents": page_agents()
    else: page_settings()


main()
