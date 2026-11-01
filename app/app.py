# =============================================================================
# app/app.py  —  Brain Tumor Detection & Explainable AI
# Deep Learning · Grad-CAM · TensorFlow · Streamlit
# =============================================================================
"""
Streamlit application for Brain Tumor MRI classification with Grad-CAM.

Run with: streamlit run app/app.py

Pages
-----
- Home          : Landing dashboard with feature overview & workflow.
- Predict MRI   : Upload → Predict → Grad-CAM + Confidence breakdown.
- About & Methods : Architecture, methodology, and references.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ── project root on path ──────────────────────────────────────────────────────
APP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config

# =============================================================================
# Page Configuration  (MUST be the first Streamlit call)
# =============================================================================
st.set_page_config(
    page_title="NeuroScan AI — Brain Tumor Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "# NeuroScan AI — Brain Tumor Detection\n"
            "Explainable AI Framework for MRI Classification using Custom CNN & Grad-CAM"
        ),
    },
)

# =============================================================================
# Design Tokens & Global CSS
# =============================================================================
_CSS = """
<style>
/* ── Google Font ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hide Streamlit chrome ───────────────────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── Main container ──────────────────────────────────────────────────────── */
.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
    max-width: 1280px;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080616 0%, #0d0b1f 50%, #111130 100%) !important;
    border-right: 1px solid rgba(124,77,255,0.2);
}
[data-testid="stSidebar"] .stMarkdown p { color: rgba(255,255,255,0.65); }
[data-testid="stSidebarCollapsedControl"] {
    background: rgba(124,77,255,0.15) !important;
    border-radius: 8px;
    margin: 0.4rem;
}
/* sidebar radio overrides */
[data-testid="stSidebar"] .stRadio > label {
    color: rgba(255,255,255,0.45) !important;
    font-size: 0.8rem;
}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}

/* ── Glassmorphism card ──────────────────────────────────────────────────── */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 14px 40px rgba(124,77,255,0.18);
}

/* ── Hero banner ─────────────────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0a0818 0%, #1a0f3c 40%, #0f1a3c 100%);
    border-radius: 20px;
    padding: 2.8rem 3.5rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(124,77,255,0.25);
    box-shadow: 0 24px 64px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(124,77,255,0.25) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 20%;
    width: 300px; height: 180px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,188,212,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #fff;
    line-height: 1.15;
    letter-spacing: -0.5px;
}
.hero-title span { color: #a78bfa; }
.hero-subtitle {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.6);
    margin-top: 0.6rem;
    font-weight: 300;
    max-width: 600px;
}
.hero-badge {
    display: inline-block;
    background: rgba(124,77,255,0.18);
    border: 1px solid rgba(124,77,255,0.45);
    color: #c4b5fd;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.4rem;
    margin-top: 1rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

/* ── Stat / metric cards ─────────────────────────────────────────────────── */
.stat-card {
    background: linear-gradient(135deg, rgba(124,77,255,0.08) 0%, rgba(0,188,212,0.05) 100%);
    border: 1px solid rgba(124,77,255,0.2);
    border-radius: 14px;
    padding: 1.3rem 1.1rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    position: relative;
    overflow: hidden;
}
.stat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(124,77,255,0.25);
    border-color: rgba(124,77,255,0.45);
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #7c4dff, #00bcd4);
}
.stat-value {
    font-size: 1.3rem;
    font-weight: 800;
    white-space: nowrap;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-label {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.5);
    margin-top: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}
.stat-icon {
    font-size: 1.6rem;
    margin-bottom: 0.4rem;
    display: block;
}

/* ── Feature card (home page) ────────────────────────────────────────────── */
.feature-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    height: 100%;
    transition: border-color 0.25s, transform 0.25s;
}
.feature-card:hover {
    border-color: rgba(124,77,255,0.4);
    transform: translateY(-3px);
}
.feature-icon {
    font-size: 2rem;
    margin-bottom: 0.6rem;
    display: block;
}
.feature-title {
    font-size: 0.98rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.4rem;
}
.feature-desc {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.5);
    line-height: 1.55;
}

/* ── Section header ──────────────────────────────────────────────────────── */
.section-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: rgba(255,255,255,0.92);
    margin-bottom: 0.9rem;
    padding-bottom: 0.45rem;
    border-bottom: 2px solid rgba(124,77,255,0.5);
    display: inline-block;
}

/* ── Page header ─────────────────────────────────────────────────────────── */
.page-header {
    margin-bottom: 1.5rem;
}
.page-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: #fff;
    margin: 0;
    letter-spacing: -0.3px;
}
.page-subtitle {
    font-size: 0.88rem;
    color: rgba(255,255,255,0.45);
    margin-top: 0.3rem;
}

/* ── Upload box ──────────────────────────────────────────────────────────── */
.upload-hint {
    text-align: center;
    color: rgba(255,255,255,0.35);
    font-size: 0.83rem;
    padding: 0.5rem;
    margin-top: -0.5rem;
}

/* ── Prediction result box ───────────────────────────────────────────────── */
.prediction-box {
    border-radius: 18px;
    padding: 2rem;
    margin: 1.2rem 0;
    border: 2px solid;
    animation: fadeInUp 0.5s ease both;
}

/* ── Probability bar ─────────────────────────────────────────────────────── */
.prob-bar-wrap {
    margin-bottom: 0.85rem;
}
.prob-label-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    margin-bottom: 0.3rem;
}
.prob-bar-outer {
    background: rgba(255,255,255,0.06);
    border-radius: 20px;
    height: 8px;
    overflow: hidden;
}
.prob-bar-inner {
    height: 100%;
    border-radius: 20px;
    transition: width 0.6s ease;
}

/* ── Confidence badge ────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 0.22rem 0.7rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-left: 0.7rem;
    vertical-align: middle;
}
.badge-green  { background: rgba(34,197,94,0.2);  border: 1px solid rgba(34,197,94,0.5);  color: #4ade80; }
.badge-orange { background: rgba(251,146,60,0.2); border: 1px solid rgba(251,146,60,0.5); color: #fb923c; }
.badge-red    { background: rgba(248,113,113,0.2);border: 1px solid rgba(248,113,113,0.5);color: #f87171; }

/* ── Info / advisory / error boxes ──────────────────────────────────────── */
.advisory-box {
    background: linear-gradient(135deg, rgba(251,146,60,0.12) 0%, rgba(0,0,0,0) 100%);
    border: 1.5px solid rgba(251,146,60,0.45);
    border-radius: 14px;
    padding: 1.1rem 1.6rem;
    margin: 1rem 0;
    animation: fadeInUp 0.4s ease both;
}
.error-box {
    background: linear-gradient(135deg, rgba(248,113,113,0.12) 0%, rgba(0,0,0,0) 100%);
    border: 1.5px solid rgba(248,113,113,0.45);
    border-radius: 14px;
    padding: 1.4rem 1.8rem;
    margin: 1.4rem 0;
    text-align: center;
    animation: fadeInUp 0.4s ease both;
}

/* ── Image panel captions ────────────────────────────────────────────────── */
.img-caption {
    text-align: center;
    font-size: 0.72rem;
    color: rgba(255,255,255,0.4);
    margin-top: 0.35rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    font-weight: 600;
}

/* ── Disclaimer pill ─────────────────────────────────────────────────────── */
.disclaimer {
    background: rgba(253,224,71,0.07);
    border: 1px solid rgba(253,224,71,0.22);
    border-radius: 10px;
    padding: 0.75rem 1.1rem;
    font-size: 0.78rem;
    color: rgba(253,224,71,0.85);
    margin-top: 1.6rem;
    line-height: 1.55;
}

/* ── Pipeline step ───────────────────────────────────────────────────────── */
.pipeline-step {
    background: linear-gradient(135deg, rgba(124,77,255,0.12), rgba(0,188,212,0.08));
    border: 1px solid rgba(124,77,255,0.3);
    border-radius: 10px;
    padding: 0.55rem 1rem;
    text-align: center;
    font-size: 0.8rem;
    color: rgba(255,255,255,0.85);
    width: 100%;
}
.pipeline-arrow { color: #7c4dff; font-size: 1.1rem; text-align: center; }

/* ── Sidebar spec pill ────────────────────────────────────────────────────── */
.spec-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.3rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.8rem;
}
.spec-key { color: rgba(255,255,255,0.45); }
.spec-val { color: #c4b5fd; font-weight: 600; font-size: 0.78rem; }

/* ── Step card ────────────────────────────────────────────────────────────── */
.step-card {
    background: rgba(124,77,255,0.06);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.step-num {
    background: rgba(124,77,255,0.3);
    color: #c4b5fd;
    border-radius: 50%;
    width: 22px; height: 22px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
}

/* ── Image details table ──────────────────────────────────────────────────── */
.img-detail-row {
    display: flex;
    justify-content: space-between;
    padding: 0.35rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 0.82rem;
}
.img-detail-key { color: rgba(255,255,255,0.45); }
.img-detail-val { color: rgba(255,255,255,0.85); font-weight: 500; }

/* ── Gradcam insight card ─────────────────────────────────────────────────── */
.insight-card {
    background: rgba(0,188,212,0.06);
    border: 1px solid rgba(0,188,212,0.2);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-top: 1.2rem;
}
.insight-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: #67e8f9;
    margin-bottom: 0.5rem;
}
.insight-body {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.65);
    line-height: 1.65;
}

/* ── Heatmap opacity slider label ─────────────────────────────────────────── */
.slider-label {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.5);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}

/* ── Animations ───────────────────────────────────────────────────────────── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position:  200% 0; }
}

/* ── Streamlit overrides ──────────────────────────────────────────────────── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #7c4dff, #00bcd4) !important;
    border-radius: 20px;
}
hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin: 1.5rem 0;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
    padding: 0.2rem;
    gap: 0.2rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: rgba(255,255,255,0.5);
    font-weight: 500;
    font-size: 0.88rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(124,77,255,0.25) !important;
    color: #c4b5fd !important;
}
div[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.02) !important;
}
div[data-testid="stExpander"]:hover {
    border-color: rgba(124,77,255,0.35) !important;
}
/* ── Metrics font size override ───────────────────────────────────────────── */
[data-testid="stMetricValue"] {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    white-space: normal !important;
    word-break: break-word !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    color: rgba(255,255,255,0.6) !important;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# =============================================================================
# Helpers
# =============================================================================
def _html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def get_class_color(class_name: str) -> str:
    from src.utils.helpers import get_class_color_map
    return get_class_color_map(Config.CLASS_NAMES).get(class_name, "#7c4dff")


def get_class_icon(class_name: str) -> str:
    return ""


def get_class_badge_class(class_name: str, is_confident: bool) -> str:
    if not is_confident:
        return "badge-orange"
    n = class_name.lower().replace("_", "")
    if "notumor" in n or "normal" in n:
        return "badge-green"
    return "badge-red"


def _load_scalar_metrics() -> dict:
    path = Config.METRICS_DIR / "scalar_metrics.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


# =============================================================================
# Model Loading (cached)
# =============================================================================
@st.cache_resource(show_spinner="Loading model — please wait...")
def load_classifier():
    """
    Load the TumorClassifier once and cache for the server lifetime.
    Returns (classifier, error_str).  Never raises to the Streamlit layer.
    """
    model_path = Config.BEST_MODEL_PATH
    if not model_path.exists():
        return None, (
            f"Model file not found at:\n{model_path}\n\n"
            "Run training first:  python main.py --stage train"
        )
    try:
        import importlib
        import src.deployment.inference as _inf
        importlib.reload(_inf)
        clf = _inf.TumorClassifier(model_path=model_path)
        return clf, None
    except FileNotFoundError as exc:
        return None, f"Model file missing: {exc}"
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _safe_predict(classifier, image_bytes: bytes, confidence_threshold: float):
    """
    Backward-compatible prediction call.
    Handles both updated and cached legacy instances gracefully.
    """
    try:
        result = classifier.predict_from_bytes(
            image_bytes, confidence_threshold=confidence_threshold
        )
    except TypeError:
        # Cached pre-update instance — patch threshold and call without kwarg
        setattr(classifier, "confidence_threshold", confidence_threshold)
        result = classifier.predict_from_bytes(image_bytes)
    # Always align is_confident with the user's active slider value
    result.is_confident = result.confidence >= confidence_threshold
    return result


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    _html("""
    <div style="padding: 1.4rem 0.5rem 0.8rem; text-align:center;">
        <div style="font-size:1.55rem; font-weight:800; color:#fff; letter-spacing:-0.5px;">
            NeuroScan AI
        </div>
        <div style="font-size:0.75rem; color:rgba(255,255,255,0.5); margin-top:0.25rem;
                    font-weight:600; text-transform:uppercase; letter-spacing:0.08em;">
            Brain Tumor Detection
        </div>
        <div style="height:1px; background:linear-gradient(90deg,transparent,rgba(124,77,255,0.5),transparent);
                    margin:1rem 0;"></div>
    </div>
    """)

    page = st.radio(
        "nav",
        options=["Home", "Predict MRI", "About & Methods"],
        label_visibility="hidden",
    )

    _html("""
    <div style="height:1px; background:linear-gradient(90deg,transparent,rgba(124,77,255,0.3),transparent);
                margin:1rem 0;"></div>
    <div style="font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em;
                color:rgba(255,255,255,0.3); font-weight:700; margin-bottom:0.6rem;">
        System Specs
    </div>
    """)

    specs = [
        ("Model",         "Custom CNN"),
        ("Input Size",    "224 × 224 RGB"),
        ("Output",        "4-Class Softmax"),
        ("Framework",     "TensorFlow"),
        ("XAI Method",    "Grad-CAM"),
        ("Threshold",     "50% (default)"),
    ]
    rows = "".join(
        f'<div class="spec-row">'
        f'<span class="spec-key">{k}</span>'
        f'<span class="spec-val">{v}</span>'
        f'</div>'
        for k, v in specs
    )
    _html(f'<div style="padding: 0 0.2rem;">{rows}</div>')

    metrics = _load_scalar_metrics()
    if metrics:
        auc = metrics.get("auc_macro", 0)
        acc = metrics.get("accuracy", 0)
        _html(f"""
        <div style="margin-top:1rem; background:rgba(124,77,255,0.08);
                    border:1px solid rgba(124,77,255,0.2); border-radius:10px;
                    padding:0.8rem 0.9rem;">
            <div style="font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em;
                        color:rgba(255,255,255,0.3); font-weight:700; margin-bottom:0.5rem;">
                Evaluation
            </div>
            <div class="spec-row">
                <span class="spec-key">Test AUC</span>
                <span class="spec-val">{auc:.4f}</span>
            </div>
            <div class="spec-row">
                <span class="spec-key">Test Acc</span>
                <span class="spec-val">{acc:.1%}</span>
            </div>
        </div>
        """)


# =============================================================================
# HOME PAGE
# =============================================================================
def render_home():
    # ── Hero ──────────────────────────────────────────────────────────────────
    _html("""
    <div class="hero-banner">
        <div class="hero-title">
            Brain Tumor Detection<br>
            <span>with Explainable AI</span>
        </div>
        <div class="hero-subtitle">
            A deep learning decision-support system for automated MRI classification,
            powered by a Custom CNN and Grad-CAM visual explanations.
        </div>
        <div>
            <span class="hero-badge">Custom CNN</span>
            <span class="hero-badge">Grad-CAM XAI</span>
            <span class="hero-badge">4-Class MRI</span>
            <span class="hero-badge">TensorFlow</span>
        </div>
    </div>
    """)

    # ── Stats row ──────────────────────────────────────────────────────────────
    metrics = _load_scalar_metrics()
    c1, c2, c3, c4, c5 = st.columns(5)
    defs = [
        (c1, "4",        "Tumor Classes"),
        (c2, "224²",     "Input Pixels"),
        (c3, "5-Block",  "CNN Depth"),
        (c4, "Grad-CAM", "XAI Method"),
        (c5, f"{metrics.get('auc_macro', 0):.3f}" if metrics else "—", "ROC-AUC"),
    ]
    for col, val, lbl in defs:
        with col:
            _html(f"""
            <div class="stat-card">
                <div class="stat-value">{val}</div>
                <div class="stat-label">{lbl}</div>
            </div>
            """)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature cards + Pipeline ──────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        _html('<p class="section-header">Key Capabilities</p>')
        fa, fb = st.columns(2)
        features = [
            ("MRI Classification",   "Classifies 224×224 RGB MRI scans into 4 tumor categories using a custom deep CNN."),
            ("Grad-CAM Localization", "Highlights the exact anatomical regions that drove the model's classification decision."),
            ("Confidence Analysis",   "Softmax probability breakdown across all classes with dynamic threshold control."),
            ("Custom CNN",            "5-block architecture with Batch Normalization, GAP, and Dropout — no pretrained weights."),
        ]
        for col, (title, desc) in zip([fa, fb, fa, fb], features):
            with col:
                _html(f"""
                <div class="feature-card">
                    <div class="feature-title">{title}</div>
                    <div class="feature-desc">{desc}</div>
                </div>
                <br>
                """)

        _html('<p class="section-header">Tumor Categories</p>')
        class_configs = {
            "glioma_tumor":     ("#ef4444", "Glioma Tumor",     "Malignant tumors arising from glial cells. Most common primary brain tumors. Include glioblastoma (GBM) and astrocytoma. Highly aggressive with infiltrative growth patterns."),
            "meningioma_tumor": ("#3b82f6", "Meningioma Tumor", "Usually benign tumors of the meninges. Slow-growing and often asymptomatic. Account for ~30% of primary brain tumors. Surgical resection is the primary treatment."),
            "pituitary_tumor":  ("#f97316", "Pituitary Tumor",  "Tumors of the pituitary gland. Usually benign adenomas. Can disrupt hormone regulation and vision. Often treated with surgery or radiation therapy."),
            "no_tumor":         ("#22c55e", "No Tumor",         "Normal brain MRI with no detectable tumor pathology. Structural anatomy appears within normal limits. No mass effect or abnormal enhancement."),
        }
        for cls_key, (color, label, desc) in class_configs.items():
            with st.expander(label):
                _html(f"""
                <div style="border-left:3px solid {color}; padding-left:1rem;
                            font-size:0.88rem; color:rgba(255,255,255,0.7);
                            line-height:1.65;">
                    {desc}
                </div>
                """)

    with col_right:
        _html('<p class="section-header">How to Use</p>')
        steps = [
            ("1", "Open the Predict MRI page from the sidebar."),
            ("2", "Upload a brain MRI scan (JPG, PNG, or BMP)."),
            ("3", "Adjust the Confidence Threshold if needed."),
            ("4", "Click Analyze Scan to run the model."),
            ("5", "Review the predicted class, probability distribution, and Grad-CAM heatmap."),
        ]
        for num, text in steps:
            _html(f"""
            <div class="step-card">
                <span class="step-num">{num}</span>
                <span style="color:rgba(255,255,255,0.8); font-size:0.87rem;">{text}</span>
            </div>
            """)

        st.markdown("<br>", unsafe_allow_html=True)
        _html('<p class="section-header">Processing Pipeline</p>')
        pipe = [
            "Brain MRI Upload (224×224×3)",
            "Preprocessing & Normalization",
            "Custom CNN Feature Extraction",
            "4-Class Softmax Classification",
            "Grad-CAM Heatmap Generation",
            "Diagnostic Report Display",
        ]
        for step in pipe:
            _html(f'<div class="pipeline-step">{step}</div>')
            if step != pipe[-1]:
                _html('<div class="pipeline-arrow">▼</div>')

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    _html("""
    <div class="disclaimer">
        <b>Medical Disclaimer:</b> This tool is <b>NOT intended for clinical medical diagnosis</b>.
        All medical decisions must be made by qualified healthcare professionals.
    </div>
    """)


# =============================================================================
# PREDICT MRI PAGE
# =============================================================================
def render_prediction():
    _html("""
    <div class="page-header">
        <div class="page-title">MRI Brain Scan Analysis</div>
        <div class="page-subtitle">
            Upload a brain MRI scan · Automated classification · Grad-CAM visual explanation
        </div>
    </div>
    """)

    classifier, model_error = load_classifier()

    # ── No model error ─────────────────────────────────────────────────────────
    if classifier is None:
        _html(f"""
        <div class="error-box">
            <div style="font-size:2rem; margin-bottom:0.5rem;">⚠️</div>
            <div style="font-size:1.3rem; font-weight:700; color:#f87171; margin-bottom:0.4rem;">
                No Trained Model Found
            </div>
            <div style="font-size:0.9rem; color:rgba(255,255,255,0.6); margin-bottom:1.2rem;">
                Please run model training first before using the prediction feature.
            </div>
            <code style="background:rgba(0,0,0,0.4); border:1px solid rgba(255,255,255,0.1);
                         border-radius:8px; padding:0.5rem 1rem; color:#a78bfa; font-size:0.9rem;">
                python main.py --stage train
            </code>
        </div>
        """)
        return

    # ── Layout: Upload | Settings ──────────────────────────────────────────────
    col_upload, col_settings = st.columns([2, 1], gap="large")

    with col_upload:
        uploaded_file = st.file_uploader(
            label="Upload Brain MRI Image",
            type=["jpg", "jpeg", "png", "bmp"],
            help="T1/T2-weighted brain MRI scan. Supported formats: JPG, JPEG, PNG, BMP.",
        )
        if not uploaded_file:
            _html('<p class="upload-hint">Drag & drop or browse for a brain MRI image (JPG / PNG / BMP)</p>')

    with col_settings:
        _html('<p class="section-header">Analysis Settings</p>')
        show_heatmap    = st.toggle("Show Grad-CAM Heatmap",        value=True,  key="tog_heatmap")
        show_all_probs  = st.toggle("Show Probability Distribution", value=True,  key="tog_probs")
        target_class    = st.selectbox(
            "Grad-CAM Target Class",
            options=["Predicted (Auto)"] + Config.CLASS_NAMES,
            help="Force Grad-CAM to highlight regions for a specific class.",
            key="sel_target",
        )
        # ── PRESERVED: interactive confidence threshold slider (50% default) ──
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.25,
            max_value=0.95,
            value=float(Config.CONFIDENCE_THRESHOLD),   # 0.50
            step=0.05,
            format="%.0f%%",
            help=(
                "Minimum softmax confidence for high-certainty label. "
                "4-class random baseline = 25%%. Calibrated default = 50%%."
            ),
            key="sld_conf",
        )
        _html(f"""
        <div style="font-size:0.75rem; color:rgba(255,255,255,0.35); margin-top:-0.4rem;">
            Random baseline: 25% &nbsp;|&nbsp; Current: <b style="color:#a78bfa;">{confidence_threshold:.0%}</b>
        </div>
        """)

    if uploaded_file is None:
        return

    # ── Display uploaded image & metadata ──────────────────────────────────────
    image_bytes = uploaded_file.read()
    pil_img     = Image.open(uploaded_file)

    st.markdown("---")

    col_img, col_meta = st.columns([1, 1], gap="large")
    with col_img:
        _html('<p class="section-header">Uploaded MRI</p>')
        st.image(image_bytes, caption=uploaded_file.name, use_container_width=True)

    with col_meta:
        _html('<p class="section-header">Image Details</p>')
        details = [
            ("Filename",    uploaded_file.name),
            ("Resolution",  f"{pil_img.size[0]} × {pil_img.size[1]} px"),
            ("Color Mode",  pil_img.mode),
            ("File Size",   f"{len(image_bytes)/1024:.1f} KB"),
            ("Target Input","224 × 224 × 3 RGB"),
        ]
        rows_html = "".join(
            f'<div class="img-detail-row">'
            f'<span class="img-detail-key">{k}</span>'
            f'<span class="img-detail-val">{v}</span>'
            f'</div>'
            for k, v in details
        )
        _html(f'<div class="glass-card" style="padding:1rem 1.2rem;">{rows_html}</div>')

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button(
            "Analyze Scan",
            type="primary",
            use_container_width=True,
            key="btn_analyze",
        )

    st.markdown("---")

    # ── Prediction ─────────────────────────────────────────────────────────────
    if not predict_btn:
        return

    with st.spinner("Analyzing MRI — running inference & Grad-CAM..."):
        class_idx = None
        if target_class != "Predicted (Auto)":
            class_idx = Config.CLASS_NAMES.index(target_class)

        # ── PRESERVED: backward-compatible call + always-aligned is_confident ──
        result = _safe_predict(classifier, image_bytes, confidence_threshold)

    # ── PRESERVED: Invalid MRI alert — hard stop ───────────────────────────────
    if not getattr(result, "is_mri", True):
        _html("""
        <div class="error-box">
            <div style="font-size:1.2rem; font-weight:700; color:#f87171; margin-bottom:0.5rem;">
                Invalid Input Image
            </div>
            <div style="font-size:0.9rem; color:rgba(255,255,255,0.7); margin-bottom:0.6rem; max-width:480px; margin-inline:auto;">
                The uploaded image did not pass brain MRI validation filters.
                Please upload a valid T1/T2-weighted brain MRI scan.
            </div>
            <div style="font-size:0.78rem; color:rgba(255,255,255,0.4);">
                Safety filter: non-medical color / structural pattern detected.
            </div>
        </div>
        """)
        return

    # ── PRESERVED: Low certainty advisory — NON-BLOCKING ─────────────────────
    is_confident = getattr(result, "is_confident", True)
    if not is_confident:
        _html(f"""
        <div class="advisory-box">
            <div style="font-size:1.05rem; font-weight:700; color:#fb923c; margin-bottom:0.4rem;">
                Low Certainty Advisory
            </div>
            <div style="font-size:0.9rem; color:rgba(255,255,255,0.85); line-height:1.55;">
                Top candidate is <b>{result.class_label}</b> with
                <b>{result.confidence:.1%}</b> confidence,
                which is below the configured threshold of
                <b>{confidence_threshold:.0%}</b>.
            </div>
            <div style="font-size:0.78rem; color:rgba(255,255,255,0.5); margin-top:0.5rem;">
                Full prediction, probability distribution, and
                Grad-CAM heatmap are still displayed below.
                Consider adjusting the Confidence Threshold in Settings.
            </div>
        </div>
        """)

    # ── Override Grad-CAM for a user-selected class ───────────────────────────
    if target_class != "Predicted (Auto)" and class_idx != result.class_idx:
        with st.spinner("Generating Grad-CAM for selected class..."):
            try:
                from src.explainability.gradcam import GradCAM as _GradCAM
                from src.preprocessing.image_utils import bytes_to_array
                cam       = _GradCAM(classifier.model)
                img_arr   = bytes_to_array(image_bytes, expand_dims=True)
                hm        = cam.compute_heatmap(img_arr, class_idx=class_idx)
                orig      = (img_arr.squeeze() * 255).astype(np.uint8)
                ov        = cam.overlay_on_image(orig, hm)
                result.gradcam_heatmap = hm
                result.gradcam_overlay = ov
            except Exception as _e:
                st.warning(f"Grad-CAM re-computation failed: {_e}")

    # ==========================================================================
    # Prediction Result Card
    # ==========================================================================
    color      = get_class_color(result.class_name)
    icon       = get_class_icon(result.class_name)
    conf_pct   = result.confidence * 100
    badge_cls  = get_class_badge_class(result.class_name, is_confident)
    badge_lbl  = "Low Certainty" if not is_confident else (
                 "No Tumor Detected" if "notumor" in result.class_name.lower().replace("_","")
                 else "Tumor Detected")

    box_bg    = f"linear-gradient(135deg, {color}18, rgba(0,0,0,0))"
    box_border = f"{color}66"

    _html(f"""
    <div class="prediction-box" style="background:{box_bg}; border-color:{box_border};">
        <div style="display:flex; align-items:flex-start; justify-content:space-between; flex-wrap:wrap; gap:1rem;">
            <div>
                <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em;
                            color:rgba(255,255,255,0.4); font-weight:700; margin-bottom:0.3rem;">
                    Predicted Diagnosis
                </div>
                <div style="font-size:1.45rem; font-weight:800; color:{color}; display:flex;
                            align-items:center; gap:0.5rem; flex-wrap:wrap;">
                    {result.class_label}
                    <span class="badge {badge_cls}">{badge_lbl}</span>
                </div>
                <div style="font-size:0.85rem; color:rgba(255,255,255,0.55); margin-top:0.35rem;">
                    Confidence: <b style="color:{color};">{conf_pct:.2f}%</b>
                    &nbsp;|&nbsp; Threshold: {confidence_threshold:.0%}
                </div>
            </div>
            <div style="text-align:right; min-width:100px;">
                <div style="font-size:2.2rem; font-weight:900; color:{color};
                            line-height:1; letter-spacing:-1px;">
                    {conf_pct:.0f}<span style="font-size:1.1rem;">%</span>
                </div>
                <div style="font-size:0.7rem; color:rgba(255,255,255,0.35);
                            text-transform:uppercase; letter-spacing:0.06em;">
                    Softmax Score
                </div>
            </div>
        </div>
        <div style="margin-top:1.2rem; font-size:0.84rem; color:rgba(255,255,255,0.65);
                    line-height:1.6; border-top:1px solid rgba(255,255,255,0.06);
                    padding-top:0.8rem;">
            {result.description}
        </div>
    </div>
    """)

    # Confidence progress bar
    _html(f"""
    <div style="margin: 0.6rem 0 0.2rem; font-size:0.75rem; color:rgba(255,255,255,0.4);
                text-transform:uppercase; letter-spacing:0.07em; font-weight:600;">
        Confidence Level
    </div>
    """)
    st.progress(min(result.confidence, 1.0))

    st.markdown("---")

    # ==========================================================================
    # Grad-CAM  +  Probability Distribution
    # ==========================================================================
    col_cam, col_probs = st.columns([3, 2], gap="large")

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    with col_cam:
        if show_heatmap:
            _html('<p class="section-header">Grad-CAM Heatmap Analysis</p>')

            import matplotlib.pyplot as _plt
            hm_colored  = _plt.cm.jet(result.gradcam_heatmap)
            hm_rgb      = (hm_colored[:, :, :3] * 255).astype(np.uint8)

            ic1, ic2, ic3 = st.columns(3)
            with ic1:
                st.image(result.original_image, use_container_width=True)
                _html('<div class="img-caption">Original MRI</div>')
            with ic2:
                st.image(hm_rgb, use_container_width=True)
                _html('<div class="img-caption">Grad-CAM Heatmap</div>')
            with ic3:
                st.image(result.gradcam_overlay, use_container_width=True)
                _html('<div class="img-caption">Overlay</div>')

            # Explainability insight card
            activation_hint = "High activation concentrated in the lesion boundary region."
            _html(f"""
            <div class="insight-card">
                <div class="insight-title">Explainability Insights</div>
                <div class="insight-body">
                    <b>What is Grad-CAM?</b><br>
                    Gradient-weighted Class Activation Mapping (Grad-CAM) visualises which
                    regions of the input MRI most influenced the CNN's prediction by back-propagating
                    gradients through the last convolutional layer.<br><br>
                    <b>Heatmap Legend:</b> Red = highest model activation
                    &nbsp;|&nbsp; Blue = lowest activation<br><br>
                    <b>Predicted class:</b> <span style="color:{color};">{result.class_label}</span>
                    &nbsp;|&nbsp; <b>Target layer:</b> last_conv_layer (Block 5)<br>
                    <span style="color:rgba(255,255,255,0.4); font-size:0.78rem;">
                        Grad-CAM supports interpretation and is not a substitute for
                        professional radiological assessment.
                    </span>
                </div>
            </div>
            """)

    # ── Probability Distribution ───────────────────────────────────────────────
    with col_probs:
        if show_all_probs:
            _html('<p class="section-header">Class Probabilities</p>')

            sorted_probs = sorted(
                result.all_probabilities.items(), key=lambda x: x[1], reverse=True
            )
            for cls_name, prob in sorted_probs:
                cls_color  = get_class_color(cls_name)
                cls_label  = Config.CLASS_LABELS.get(cls_name, cls_name)
                cls_icon   = ""
                is_pred    = cls_name == result.class_name
                label_style = 'color:#fff; font-weight:700;' if is_pred else 'color:rgba(255,255,255,0.6);'
                pred_tag   = ' <span style="font-size:0.7rem; color:#a78bfa;">(Predicted)</span>' if is_pred else ''
                pct_w      = max(int(prob * 100), 2)

                _html(f"""
                <div class="prob-bar-wrap">
                    <div class="prob-label-row">
                        <span style="{label_style}">{cls_label}{pred_tag}</span>
                        <span style="color:{cls_color}; font-weight:700;">{prob:.1%}</span>
                    </div>
                    <div class="prob-bar-outer">
                        <div class="prob-bar-inner"
                             style="width:{pct_w}%; background:{cls_color};
                                    box-shadow: 0 0 6px {cls_color}88;">
                        </div>
                    </div>
                </div>
                """)

            # Summary cards
            st.markdown("<br>", unsafe_allow_html=True)
            _html(f"""
            <p class="section-header">Prediction Summary</p>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; margin-top: 0.3rem;">
                <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
                            border-radius: 10px; padding: 0.75rem 0.8rem;">
                    <div style="font-size: 0.72rem; color: rgba(255,255,255,0.5); text-transform: uppercase;
                                letter-spacing: 0.05em; font-weight: 600; margin-bottom: 0.25rem;">
                        Tumor Type
                    </div>
                    <div style="font-size: 0.98rem; font-weight: 700; color: #fff; line-height: 1.25;">
                        {result.class_label}
                    </div>
                </div>
                <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
                            border-radius: 10px; padding: 0.75rem 0.8rem;">
                    <div style="font-size: 0.72rem; color: rgba(255,255,255,0.5); text-transform: uppercase;
                                letter-spacing: 0.05em; font-weight: 600; margin-bottom: 0.25rem;">
                        Confidence
                    </div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #a78bfa; line-height: 1.25;">
                        {result.confidence:.2%}
                    </div>
                </div>
                <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
                            border-radius: 10px; padding: 0.75rem 0.8rem;">
                    <div style="font-size: 0.72rem; color: rgba(255,255,255,0.5); text-transform: uppercase;
                                letter-spacing: 0.05em; font-weight: 600; margin-bottom: 0.25rem;">
                        Uncertainty
                    </div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #60a5fa; line-height: 1.25;">
                        {1 - result.confidence:.2%}
                    </div>
                </div>
            </div>
            """)

    # Disclaimer
    _html("""
    <div class="disclaimer">
        <b>Disclaimer:</b> This prediction must <b>NOT</b> be used as a substitute for professional medical diagnosis.
    </div>
    """)


# =============================================================================
# ABOUT & METHODS PAGE
# =============================================================================
def render_about():
    _html("""
    <div class="page-header">
        <div class="page-title">System Architecture & Methods</div>
        <div class="page-subtitle">
            Technical documentation — CNN design, training methodology, and Grad-CAM formulation
        </div>
    </div>
    """)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Architecture",
        "Methodology",
        "Grad-CAM",
        "Evaluation",
        "References",
    ])

    # ── Tab 1: Architecture ────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Custom CNN Architecture")
        st.info(
            "This model was built from scratch — no pretrained weights (ImageNet, etc.) were used. "
            "Every layer was designed specifically for brain MRI classification."
        )

        col_a, col_b = st.columns([3, 2], gap="large")
        with col_a:
            st.markdown("""
| Layer Block | Output Shape | Key Components |
|---|---|---|
| Input | 224 × 224 × 3 | RGB MRI |
| Block 1: Conv(32)×2 + MaxPool | 112 × 112 × 32 | BN + ReLU |
| Block 2: Conv(64)×2 + MaxPool | 56 × 56 × 64 | BN + ReLU |
| Block 3: Conv(128)×2 + MaxPool | 28 × 28 × 128 | BN + ReLU |
| Block 4: Conv(256)×2 + MaxPool | 14 × 14 × 256 | BN + ReLU |
| **Block 5: Conv(256)×2** *(Grad-CAM target)* | 14 × 14 × 256 | No pooling |
| Global Average Pooling | 256 | Replaces Flatten |
| Dense(512) + BN + Dropout(0.4) | 512 | — |
| Dense(256) + BN + Dropout(0.25) | 256 | — |
| Dense(4) + Softmax | **4** | Classification |
""")

        with col_b:
            st.markdown("#### Design Decisions")
            decisions = [
                ("Batch Normalization", "After every Conv layer — stabilises training, acts as regulariser."),
                ("Global Avg Pool", "Replaces Flatten — fewer parameters, less overfitting."),
                ("Block 5 No Pooling", "Preserves spatial resolution for detailed Grad-CAM heatmaps."),
                ("Label Smoothing α=0.05", "Prevents overconfidence; calibrates softmax probabilities."),
                ("He Normal Init", "Optimal weight initialisation for ReLU networks."),
                ("Gradient Clipping", "clipnorm=1.0 prevents training instability."),
            ]
            for title, body in decisions:
                with st.expander(title):
                    st.markdown(f"<span style='color:rgba(255,255,255,0.7);font-size:0.88rem;'>{body}</span>", unsafe_allow_html=True)

        st.markdown("#### CNN Architecture Flow")
        blocks = [
            ("Input",      "224×224×3"),
            ("Block 1",    "32 filters"),
            ("Block 2",    "64 filters"),
            ("Block 3",    "128 filters"),
            ("Block 4",    "256 filters"),
            ("Block 5",    "Grad-CAM layer"),
            ("GAP → Dense","4-class Softmax"),
        ]
        cols = st.columns(len(blocks))
        for col, (title, sub) in zip(cols, blocks):
            with col:
                is_gcam = "Grad-CAM" in sub
                bdr = "#ef4444" if is_gcam else "rgba(124,77,255,0.4)"
                bg  = "rgba(239,68,68,0.12)" if is_gcam else "rgba(124,77,255,0.1)"
                _html(f"""
                <div style="background:{bg}; border:1px solid {bdr}; border-radius:10px;
                            padding:0.5rem; text-align:center; font-size:0.78rem;">
                    <b style="color:{'#fca5a5' if is_gcam else '#c4b5fd'};">{title}</b><br>
                    <span style="font-size:0.68rem; color:rgba(255,255,255,0.45);">{sub}</span>
                </div>
                """)

    # ── Tab 2: Methodology ─────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Development Methodology")

        col_m1, col_m2 = st.columns(2, gap="large")
        with col_m1:
            st.markdown("""
#### Dataset
- **Source**: Brain Tumor MRI Dataset (Kaggle)
- **Classes**: Glioma · Meningioma · No Tumor · Pituitary
- **Split**: Training (80%) · Validation (20%) · Testing (held-out)

#### Preprocessing
1. Resize all images to **224×224 pixels**
2. Convert to **3-channel RGB** (handles grayscale MRIs)
3. Normalise pixel values to **[0, 1]**
4. Clip augmented values to valid range

#### Data Augmentation (Training Only)
- Random horizontal flip
- Random rotation ±15°
- Random zoom ±10%
- Random translation ±10%
- Random brightness adjustment ±20%
""")

        with col_m2:
            st.markdown("""
#### Training Configuration

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Initial LR | 3×10⁻⁴ |
| Gradient Clipping | clipnorm = 1.0 |
| Loss | Cat. Cross-entropy + LS |
| Label Smoothing | α = 0.05 |
| Batch Size | 32 |
| Max Epochs | 60 |
| Class Weights | Balanced |

#### Callbacks

| Callback | Trigger |
|---|---|
| ModelCheckpoint | Best val_accuracy |
| EarlyStopping | Patience = 12 epochs |
| ReduceLROnPlateau | Factor 0.5, patience 5 |
| CSVLogger | Per-epoch metrics |
""")

    # ── Tab 3: Grad-CAM ────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Gradient-weighted Class Activation Mapping (Grad-CAM)")

        col_g1, col_g2 = st.columns([3, 2], gap="large")
        with col_g1:
            st.markdown(r"""
Grad-CAM produces class-discriminative spatial heatmaps that explain which
regions of the MRI image influenced the model's prediction.

#### Mathematical Formulation

**Step 1 — Forward Pass:** Obtain activation maps from the last conv layer:
$$A^k \in \mathbb{R}^{H \times W}$$

**Step 2 — Gradient Computation:** Compute gradient of class score $y^c$
w.r.t. each spatial activation:
$$\frac{\partial y^c}{\partial A^k_{ij}}$$

**Step 3 — Importance Weights:** Global average pool the gradients:
$$\alpha^c_k = \frac{1}{Z} \sum_{i,j} \frac{\partial y^c}{\partial A^k_{ij}}$$

**Step 4 — Weighted Combination + ReLU:**
$$L^c_{\text{Grad-CAM}} = \text{ReLU}\!\left(\sum_k \alpha^c_k A^k\right)$$

**Step 5:** Upsample heatmap to 224×224 and overlay on MRI.
""")

        with col_g2:
            st.markdown("#### Heatmap Interpretation")
            interp = [
                ("Red / Yellow",   "Highest activation — primary region of model focus for the predicted class."),
                ("Yellow / Green", "Moderate activation — secondary contributing regions."),
                ("Blue",           "Minimal activation — less relevant to the final decision."),
            ]
            for color_lbl, desc in interp:
                _html(f"""
                <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
                            border-radius:10px; padding:0.7rem 1rem; margin-bottom:0.6rem;">
                    <b style="font-size:0.9rem;">{color_lbl}</b><br>
                    <span style="font-size:0.82rem; color:rgba(255,255,255,0.6);">{desc}</span>
                </div>
                """)

            _html("""
            <div class="insight-card" style="margin-top:0.8rem;">
                <div class="insight-title">Important Note</div>
                <div class="insight-body">
                    Grad-CAM provides explanations that assist interpretation but are
                    <b>not diagnostic conclusions</b>. Always consult a qualified
                    radiologist for clinical decisions.
                </div>
            </div>
            """)

    # ── Tab 4: Evaluation ──────────────────────────────────────────────────────
    with tab4:
        st.markdown("### Model Evaluation Results")
        metrics = _load_scalar_metrics()
        if metrics:
            mc = st.columns(4)
            eval_stats = [
                (mc[0], "Test Accuracy",    f"{metrics.get('accuracy', 0):.2%}",         "Overall correct classifications on held-out test set."),
                (mc[1], "Macro AUC",        f"{metrics.get('auc_macro', 0):.4f}",         "ROC-AUC averaged equally across all 4 classes."),
                (mc[2], "Macro F1-Score",   f"{metrics.get('macro_f1', 0):.4f}",          "Harmonic mean of precision and recall (macro)."),
                (mc[3], "Cohen's Kappa",    f"{metrics.get('kappa', 0):.4f}",             "Agreement beyond chance between model and ground truth."),
            ]
            for col, label, val, tip in eval_stats:
                with col:
                    _html(f"""
                    <div class="stat-card">
                        <div class="stat-value">{val}</div>
                        <div class="stat-label">{label}</div>
                    </div>
                    """)
                    st.caption(tip)

            st.markdown("---")
            c_l, c_r = st.columns(2)
            with c_l:
                st.markdown("#### Full Metrics Table")
                st.table({
                    "Metric":  ["Accuracy", "Macro Precision", "Macro Recall",
                                "Macro F1", "Weighted F1", "Kappa", "Macro AUC", "Weighted AUC"],
                    "Value": [
                        f"{metrics.get('accuracy',0):.4f}",
                        f"{metrics.get('macro_precision',0):.4f}",
                        f"{metrics.get('macro_recall',0):.4f}",
                        f"{metrics.get('macro_f1',0):.4f}",
                        f"{metrics.get('weighted_f1',0):.4f}",
                        f"{metrics.get('kappa',0):.4f}",
                        f"{metrics.get('auc_macro',0):.4f}",
                        f"{metrics.get('auc_weighted',0):.4f}",
                    ],
                })

            with c_r:
                # Show training history plot if available
                plot_path = Config.PLOTS_DIR / "training_history.png"
                if plot_path.exists():
                    st.markdown("#### Training History")
                    st.image(str(plot_path), use_container_width=True)

                # Confusion matrix
                cm_path = Config.CONFUSION_MATRIX_DIR / "confusion_matrix_normalized.png"
                if cm_path.exists():
                    st.markdown("#### Normalised Confusion Matrix")
                    st.image(str(cm_path), use_container_width=True)
        else:
            st.info(
                "No evaluation metrics found. Run the evaluation stage:\n"
                "`python main.py --stage evaluate`"
            )

    # ── Tab 5: References ──────────────────────────────────────────────────────
    with tab5:
        st.markdown("### Technical & Literature References")
        refs = [
            ("Selvaraju et al., 2017",   "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. *ICCV 2017*."),
            ("Ioffe & Szegedy, 2015",    "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *ICML 2015*."),
            ("Lin et al., 2013",         "Network In Network. *ICLR 2014*. (Global Average Pooling)."),
            ("Srivastava et al., 2014",  "Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR 2014*."),
            ("Kingma & Ba, 2015",        "Adam: A Method for Stochastic Optimization. *ICLR 2015*."),
            ("LeCun et al., 1998",       "Gradient-Based Learning Applied to Document Recognition. *Proc. IEEE 1998*. (CNN foundations)."),
        ]
        for authors, title in refs:
            _html(f"""
            <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06);
                        border-radius:10px; padding:0.8rem 1.1rem; margin-bottom:0.5rem;">
                <span style="color:#a78bfa; font-weight:600; font-size:0.85rem;">{authors}</span><br>
                <span style="color:rgba(255,255,255,0.65); font-size:0.83rem;">{title}</span>
            </div>
            """)

        st.markdown("---")
        st.markdown("**Dataset:** [Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)")


# =============================================================================
# Router
# =============================================================================
if page == "Home":            render_home()
elif page == "Predict MRI":   render_prediction()
elif page == "About & Methods": render_about()
