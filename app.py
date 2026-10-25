"""
HeritageAI (செம்மொழி சுடர்) — Premium Streamlit UI
Restores ancient Tamil palm leaf manuscripts using OpenCV,
reads them via Gemini 2.5 Flash REST API, and translates with gTTS.
"""

import streamlit as st
import cv2
import numpy as np
import requests
import base64
import io
import os
import tempfile
from PIL import Image
from gtts import gTTS

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HeritageAI · செம்மொழி சுடர்",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GOOGLE FONTS + GLOBAL CSS
# ─────────────────────────────────────────────
GLOBAL_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;800&family=Inter:wght@300;400;500;600&family=Mukta+Malar:wght@300;400;600&display=swap" rel="stylesheet">

<style>
/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0F172A !important;
    color: #CBD5E1 !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #1E293B; }
::-webkit-scrollbar-thumb { background: #FBBF24; border-radius: 4px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
    border-right: 1px solid #263552 !important;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FBBF24 !important;
    font-family: 'Cinzel', serif !important;
}

/* ── Slider ── */
[data-testid="stSlider"] .stSlider > div > div > div {
    background: #FBBF24 !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: #FBBF24 !important;
    border: 2px solid #F59E0B !important;
}

/* ── Select box ── */
[data-testid="stSelectbox"] > div > div {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    color: #CBD5E1 !important;
    border-radius: 8px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #1E293B !important;
    border: 2px dashed #FBBF24 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"] label { color: #FBBF24 !important; font-weight: 600 !important; }
[data-testid="stFileUploadDropzone"] small { color: #94A3B8 !important; }

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #F59E0B 0%, #FBBF24 100%) !important;
    color: #0F172A !important;
    font-family: 'Cinzel', serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2rem !important;
    border: none !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    letter-spacing: 0.08em !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    box-shadow: 0 4px 15px rgba(251, 191, 36, 0.35) !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(251, 191, 36, 0.55) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p { color: #FBBF24 !important; font-family: 'Cinzel', serif !important; }

/* ── Column gap fix ── */
[data-testid="column"] { padding: 0 0.75rem !important; }

/* ── Image captions ── */
[data-testid="stImage"] figcaption {
    color: #94A3B8 !important;
    font-size: 0.78rem !important;
    text-align: center !important;
    margin-top: 4px !important;
}

/* ── Audio player ── */
audio {
    width: 100% !important;
    border-radius: 8px !important;
    margin-top: 0.5rem !important;
}

/* ── Divider ── */
hr { border-color: #1E3A5F !important; }
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  COMPONENT HTML HELPERS
# ─────────────────────────────────────────────

HERO_HTML = """
<div style="
    text-align: center;
    padding: 3.5rem 1rem 2.5rem;
    background: radial-gradient(ellipse at 50% 0%, rgba(251,191,36,0.12) 0%, transparent 65%);
    margin-bottom: 0.5rem;
">
    <!-- Decorative top rule -->
    <div style="
        width: 80px; height: 2px;
        background: linear-gradient(90deg, transparent, #FBBF24, transparent);
        margin: 0 auto 1.6rem;
    "></div>

    <!-- Badge -->
    <span style="
        display: inline-block;
        background: rgba(251,191,36,0.12);
        border: 1px solid rgba(251,191,36,0.4);
        color: #FBBF24;
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        padding: 0.3rem 1rem;
        border-radius: 99px;
        margin-bottom: 1.3rem;
    ">
        ✦ Powered by Gemini 2.5 Flash · OpenCV · gTTS ✦
    </span>

    <!-- Main Title -->
    <h1 style="
        font-family: 'Cinzel', serif;
        font-size: clamp(2rem, 5vw, 3.4rem);
        font-weight: 800;
        color: #FBBF24;
        letter-spacing: 0.04em;
        line-height: 1.15;
        text-shadow:
            0 0 30px rgba(251,191,36,0.6),
            0 0 80px rgba(251,191,36,0.25);
        margin: 0 auto 0.6rem;
    ">
        🏛️ HeritageAI: Unlocking Ancient Wisdom
    </h1>

    <!-- Tamil Title -->
    <p style="
        font-family: 'Mukta Malar', sans-serif;
        font-size: 1.35rem;
        font-weight: 600;
        color: #F59E0B;
        margin: 0 auto 1rem;
        letter-spacing: 0.05em;
    ">செம்மொழி சுடர்</p>

    <!-- Subtitle -->
    <p style="
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        font-weight: 300;
        color: #94A3B8;
        max-width: 640px;
        margin: 0 auto 1.6rem;
        line-height: 1.7;
    ">
        Preserving Sangam Literature through Advanced Vision &amp;
        Multimodal AI — restoring the irreplaceable voices of antiquity.
    </p>

    <!-- Stat pills -->
    <div style="display:flex; justify-content:center; gap:1.2rem; flex-wrap:wrap; margin-bottom:1rem;">
        <span style="background:#1E293B;border:1px solid #334155;border-radius:8px;padding:0.45rem 1.1rem;font-size:0.82rem;color:#CBD5E1;">
            📜 Palm Leaf OCR
        </span>
        <span style="background:#1E293B;border:1px solid #334155;border-radius:8px;padding:0.45rem 1.1rem;font-size:0.82rem;color:#CBD5E1;">
            🔬 AI Image Restoration
        </span>
        <span style="background:#1E293B;border:1px solid #334155;border-radius:8px;padding:0.45rem 1.1rem;font-size:0.82rem;color:#CBD5E1;">
            🌐 Multi-language Translation
        </span>
        <span style="background:#1E293B;border:1px solid #334155;border-radius:8px;padding:0.45rem 1.1rem;font-size:0.82rem;color:#CBD5E1;">
            🔊 Audio Narration
        </span>
    </div>

    <!-- Bottom rule -->
    <div style="
        width: 120px; height: 1px;
        background: linear-gradient(90deg, transparent, #334155, transparent);
        margin: 0 auto;
    "></div>
</div>
"""


def section_label(icon: str, text: str, color: str = "#FBBF24") -> str:
    return f"""
    <div style="
        display:flex; align-items:center; gap:0.55rem;
        margin-bottom:0.85rem; margin-top:0.3rem;
    ">
        <span style="font-size:1.25rem;">{icon}</span>
        <span style="
            font-family:'Cinzel',serif;
            font-size:0.95rem;
            font-weight:600;
            color:{color};
            letter-spacing:0.08em;
            text-transform:uppercase;
        ">{text}</span>
        <div style="flex:1;height:1px;background:linear-gradient(90deg,#334155,transparent);margin-left:0.4rem;"></div>
    </div>
    """


def image_card(label: str) -> str:
    """CSS wrapper rendered above columns as a label."""
    return f"""
    <div style="
        text-align:center;
        font-family:'Inter',sans-serif;
        font-size:0.78rem;
        font-weight:600;
        letter-spacing:0.12em;
        text-transform:uppercase;
        color:#94A3B8;
        margin-bottom:0.4rem;
    ">{label}</div>
    """


def ocr_box(text: str) -> str:
    return f"""
    <div style="
        background:#0D1B2A;
        border:1.5px dashed #64748B;
        border-radius:10px;
        padding:1.3rem 1.5rem;
        margin-bottom:1.2rem;
        font-family:'Mukta Malar',sans-serif;
        font-size:1.05rem;
        line-height:1.9;
        color:#CBD5E1;
        white-space:pre-wrap;
        word-break:break-word;
    ">
        <div style="
            font-family:'Inter',sans-serif;
            font-size:0.68rem;
            font-weight:700;
            letter-spacing:0.15em;
            text-transform:uppercase;
            color:#64748B;
            margin-bottom:0.8rem;
        ">📜 Extracted Ancient Tamil</div>
        {text}
    </div>
    """


def translation_box(text: str) -> str:
    return f"""
    <div style="
        background:#1E293B;
        border:1.5px solid #FBBF24;
        border-radius:10px;
        padding:1.3rem 1.5rem;
        margin-bottom:1.2rem;
        font-family:'Inter',sans-serif;
        font-size:1rem;
        line-height:1.8;
        color:#E2E8F0;
        white-space:pre-wrap;
        word-break:break-word;
        box-shadow:0 4px 24px rgba(251,191,36,0.15), 0 1px 4px rgba(0,0,0,0.4);
    ">
        <div style="
            font-size:0.68rem;
            font-weight:700;
            letter-spacing:0.15em;
            text-transform:uppercase;
            color:#FBBF24;
            margin-bottom:0.8rem;
        ">🌐 Modern Translation</div>
        {text}
    </div>
    """


def sidebar_how_it_works() -> str:
    return """
    <div style="
        background:rgba(251,191,36,0.06);
        border:1px solid rgba(251,191,36,0.2);
        border-radius:10px;
        padding:1rem 1.1rem;
        margin-top:1.4rem;
        font-family:'Inter',sans-serif;
        font-size:0.82rem;
        line-height:1.7;
        color:#94A3B8;
    ">
        <div style="
            font-family:'Cinzel',serif;
            font-size:0.78rem;
            font-weight:600;
            color:#FBBF24;
            letter-spacing:0.1em;
            text-transform:uppercase;
            margin-bottom:0.65rem;
        ">⚙ How It Works</div>
        <ol style="margin:0;padding-left:1.1rem;">
            <li style="margin-bottom:0.35rem;"><b style="color:#E2E8F0;">Upload</b> a palm leaf scan (JPG/PNG).</li>
            <li style="margin-bottom:0.35rem;"><b style="color:#E2E8F0;">Adjust</b> the CLAHE slider for optimal contrast.</li>
            <li style="margin-bottom:0.35rem;"><b style="color:#E2E8F0;">Click</b> <em style="color:#FBBF24;">Extract &amp; Translate</em>.</li>
            <li style="margin-bottom:0.35rem;"><b style="color:#E2E8F0;">Read</b> the extracted Tamil &amp; translation.</li>
            <li><b style="color:#E2E8F0;">Listen</b> to the narrated audio output.</li>
        </ol>
    </div>
    """


# ─────────────────────────────────────────────
#  BACKEND LOGIC — OpenCV Restoration
# ─────────────────────────────────────────────

def restore_image(pil_image: Image.Image, clip_limit: float) -> np.ndarray:
    """Apply CLAHE contrast enhancement + adaptive threshold to a PIL image."""
    img_array = np.array(pil_image.convert("L"))  # grayscale
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl_img = clahe.apply(img_array)
    restored = cv2.adaptiveThreshold(
        cl_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    return restored


# ─────────────────────────────────────────────
#  BACKEND LOGIC — Gemini REST API
# ─────────────────────────────────────────────

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

LANG_CODES = {
    "English": "en",
    "Tamil (தமிழ்)": "ta",
    "Hindi (हिन्दी)": "hi",
    "French (Français)": "fr",
    "German (Deutsch)": "de",
    "Spanish (Español)": "es",
    "Japanese (日本語)": "ja",
    "Arabic (عربي)": "ar",
}


def encode_image_base64(pil_image: Image.Image) -> str:
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def call_gemini(api_key: str, image_b64: str, target_language: str) -> str:
    """Call Gemini 2.5 Flash REST endpoint and return raw text response."""
    prompt = (
        f"You are an expert in ancient Tamil epigraphy and classical literature. "
        f"Analyze this image of a palm leaf manuscript or stone inscription.\n\n"
        f"1. Extract ALL visible Tamil text exactly as written (ancient script).\n"
        f"2. Provide a detailed scholarly translation and explanation in {target_language}.\n\n"
        f"Format your response EXACTLY as:\n"
        f"[ANCIENT TEXT]\n<extracted Tamil text here>\n---SPLIT---\n"
        f"[TRANSLATION]\n<translation and explanation in {target_language} here>"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_b64,
                        }
                    },
                ]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}

    response = requests.post(
        GEMINI_API_URL,
        json=payload,
        headers=headers,
        params=params,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def parse_gemini_response(raw: str):
    """Deterministic ---SPLIT--- parser. Returns (ancient_text, translation)."""
    if "---SPLIT---" in raw:
        parts = raw.split("---SPLIT---", 1)
        ancient = parts[0].replace("[ANCIENT TEXT]", "").strip()
        translation = parts[1].replace("[TRANSLATION]", "").strip()
    else:
        # Graceful fallback
        ancient = raw.strip()
        translation = ""
    return ancient, translation


def generate_audio(text: str, lang_code: str) -> str:
    """Generate a gTTS audio file and return the temp file path."""
    tts = gTTS(text=text, lang=lang_code, slow=False)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style="
            text-align:center; padding:1.2rem 0 0.8rem;
            border-bottom:1px solid #1E3A5F; margin-bottom:1.2rem;
        ">
            <div style="font-size:2rem;">🏛️</div>
            <div style="
                font-family:'Cinzel',serif;
                font-size:1.05rem;
                font-weight:700;
                color:#FBBF24;
                letter-spacing:0.08em;
            ">HeritageAI</div>
            <div style="
                font-family:'Mukta Malar',sans-serif;
                font-size:0.82rem;
                color:#64748B;
                margin-top:2px;
            ">செம்மொழி சுடர்</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── API Key ──
    st.markdown(
        "<p style='font-family:Cinzel,serif;font-size:0.82rem;color:#FBBF24;"
        "letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.3rem;'>"
        "🔑 Gemini API Key</p>",
        unsafe_allow_html=True,
    )
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        label_visibility="collapsed",
    )

    st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)

    # ── Restoration Tools ──
    st.markdown(
        "<p style='font-family:Cinzel,serif;font-size:0.82rem;color:#FBBF24;"
        "letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.3rem;'>"
        "🔬 Restoration Tools</p>",
        unsafe_allow_html=True,
    )
    clip_limit = st.slider(
        "CLAHE Clip Limit",
        min_value=1.0,
        max_value=8.0,
        value=3.0,
        step=0.5,
        help="Higher values increase local contrast. Recommended: 2–4 for most manuscripts.",
    )

    st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)

    # ── Global Accessibility ──
    st.markdown(
        "<p style='font-family:Cinzel,serif;font-size:0.82rem;color:#FBBF24;"
        "letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.3rem;'>"
        "🌐 Global Accessibility</p>",
        unsafe_allow_html=True,
    )
    selected_language = st.selectbox(
        "Translation Language",
        options=list(LANG_CODES.keys()),
        index=0,
        label_visibility="collapsed",
    )

    # ── How it Works ──
    st.markdown(sidebar_how_it_works(), unsafe_allow_html=True)

    # ── Footer ──
    st.markdown(
        """
        <div style="
            margin-top:2rem;
            padding-top:1rem;
            border-top:1px solid #1E3A5F;
            text-align:center;
            font-family:'Inter',sans-serif;
            font-size:0.72rem;
            color:#334155;
            line-height:1.6;
        ">
            Built for the Hackathon 🏆<br>
            <span style="color:#FBBF24;">HeritageAI</span> · 2025
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown(HERO_HTML, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  UPLOAD ZONE
# ─────────────────────────────────────────────
st.markdown(section_label("📂", "Upload Manuscript"), unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag & drop a palm leaf scan — JPG, PNG, or JPEG",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  VISION COMPARISON
# ─────────────────────────────────────────────
if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)

    # Run OpenCV restoration
    restored_array = restore_image(pil_image, clip_limit)
    restored_pil = Image.fromarray(restored_array)

    st.markdown(section_label("🔬", "Vision Comparison"), unsafe_allow_html=True)

    col_orig, col_rest = st.columns(2, gap="medium")

    with col_orig:
        st.markdown(image_card("📜 Original Artifact"), unsafe_allow_html=True)
        st.image(
            pil_image,
            use_container_width=True,
            caption="Uploaded scan — unprocessed",
        )

    with col_rest:
        st.markdown(image_card("✨ AI Restored View"), unsafe_allow_html=True)
        st.image(
            restored_pil,
            use_container_width=True,
            caption=f"CLAHE ({clip_limit}×) + Adaptive Threshold",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Extract & Translate Button ──
    st.markdown(section_label("⚡", "AI Extraction & Translation"), unsafe_allow_html=True)

    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        run_extraction = st.button("⚡ Extract & Translate", use_container_width=True)

    if run_extraction:
        if not api_key:
            st.markdown(
                """
                <div style="
                    background:rgba(239,68,68,0.1);
                    border:1px solid rgba(239,68,68,0.4);
                    border-radius:8px;
                    padding:0.9rem 1.2rem;
                    color:#FCA5A5;
                    font-family:'Inter',sans-serif;
                    font-size:0.9rem;
                    text-align:center;
                    margin-top:0.8rem;
                ">
                    ⚠️ Please enter your <strong>Gemini API Key</strong> in the sidebar before running extraction.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            lang_code = LANG_CODES[selected_language]

            with st.spinner("🔍 Scanning ancient script… communicating with Gemini 2.5 Flash…"):
                try:
                    # Use the restored image for OCR (better contrast)
                    image_b64 = encode_image_base64(restored_pil)
                    raw_response = call_gemini(api_key, image_b64, selected_language)
                    ancient_text, translation = parse_gemini_response(raw_response)

                    # ── OCR Output ──
                    if ancient_text:
                        st.markdown(ocr_box(ancient_text), unsafe_allow_html=True)
                    else:
                        st.info("No ancient text could be extracted from this image.")

                    # ── Translation Output ──
                    if translation:
                        st.markdown(translation_box(translation), unsafe_allow_html=True)

                        # ── Audio ──
                        st.markdown(
                            section_label("🔊", "Audio Narration"),
                            unsafe_allow_html=True,
                        )
                        try:
                            audio_path = generate_audio(translation, lang_code)
                            with open(audio_path, "rb") as af:
                                audio_bytes = af.read()
                            st.audio(audio_bytes, format="audio/mp3")
                            os.unlink(audio_path)
                        except Exception as audio_err:
                            st.warning(f"Audio generation failed: {audio_err}")

                except requests.exceptions.HTTPError as http_err:
                    st.error(f"Gemini API error: {http_err}")
                except Exception as err:
                    st.error(f"Unexpected error: {err}")

else:
    # ── Placeholder / empty state ──
    st.markdown(
        """
        <div style="
            background:rgba(30,41,59,0.5);
            border:1px dashed #334155;
            border-radius:12px;
            padding:3.5rem 2rem;
            text-align:center;
            font-family:'Inter',sans-serif;
        ">
            <div style="font-size:3rem; margin-bottom:1rem;">📜</div>
            <div style="font-size:1.05rem; color:#64748B; font-weight:400; line-height:1.7;">
                Upload a palm leaf manuscript scan above<br>
                to begin the AI restoration &amp; analysis.
            </div>
            <div style="
                margin-top:1.4rem;
                font-family:'Mukta Malar',sans-serif;
                font-size:1rem;
                color:#334155;
            ">
                தமிழ் சிரமங்களை நாம் மீட்டெடுக்கிறோம் · We restore Tamil heritage
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  PAGE FOOTER
# ─────────────────────────────────────────────
st.markdown(
    """
    <div style="
        margin-top:4rem;
        padding:1.5rem 0;
        border-top:1px solid #1E293B;
        text-align:center;
        font-family:'Inter',sans-serif;
        font-size:0.78rem;
        color:#334155;
        line-height:1.9;
    ">
        <span style="color:#FBBF24;font-family:'Cinzel',serif;font-weight:600;">HeritageAI</span>
        · செம்மொழி சுடர் · Powered by Gemini 2.5 Flash, OpenCV &amp; gTTS<br>
        Made with ❤️ to preserve the irreplaceable legacy of Tamil civilisation.
    </div>
    """,
    unsafe_allow_html=True,
)