import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import time

# -------------------------------
# 🔹 PAGE CONFIG & THEME
# -------------------------------
st.set_page_config(page_title="Fresh Produce Quality Grading", layout="wide")

st.markdown(
    """
    <style>
    :root{
      --bg: #0f1724;
      --accent: #16a34a;
      --muted: #94a3b8;
      --glass: rgba(255,255,255,0.04);
      --glass-2: rgba(255,255,255,0.02);
      --shadow: 0 10px 30px rgba(2,6,23,0.6);
      --radius: 14px;
      font-family: Inter, sans-serif;
    }
    body, .stApp {
        background: #0f1724;
        color: #e6eef6;
    }
    button[kind="primary"],
    .stButton>button,
    .stDownloadButton>button {
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 18px;
        background: linear-gradient(90deg, #14b67a, #0ea5a4);
        color: white !important;
        border: none;
        box-shadow: 0 4px 12px rgba(20,182,122,0.25);
        transition: all 0.3s ease;
    }
    button[kind="primary"]:hover,
    .stButton>button:hover,
    .stDownloadButton>button:hover {
        transform: scale(1.07);
        box-shadow: 0 8px 25px rgba(20,182,122,0.4);
        background: linear-gradient(90deg, #16d58a, #10a5a4);
        border: 1px solid rgba(255,255,255,0.1);
        cursor: pointer;
    }
    div[data-testid="stFileUploader"] section {
        border-radius: 12px;
        border: 2px dashed rgba(255,255,255,0.2);
        background-color: rgba(255,255,255,0.02);
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"] section:hover {
        background-color: rgba(255,255,255,0.05);
        border-color: #16a34a;
        box-shadow: 0 0 12px rgba(22,163,74,0.3);
        transform: scale(1.01);
    }
    img {
        border-radius: 12px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    img:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 18px rgba(20,182,122,0.3);
    }
    .good { color: #10b981; font-weight: bold; }
    .ok { color: #f59e0b; font-weight: bold; }
    .bad { color: #ef4444; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True
)

# -------------------------------
# 🔹 HEADER
# -------------------------------
st.markdown("""
<div style="display:flex; align-items:center; gap:18px; margin-bottom:20px;">
  <div style="width:64px; height:64px; border-radius:12px; background:linear-gradient(135deg,#14b67a,#0ea5a4); display:flex; align-items:center; justify-content:center; box-shadow:0 6px 20px rgba(20,182,122,0.18);">
    <svg viewBox="0 0 24 24" fill="none" width="32" height="32"><path d="M12 3c3 0 6 1.5 7.5 4.5S19.2 12 16 14s-8 3-10 1c0 0 2-5 6-9 1.3-1.3 2.7-2 0-3z" fill="#fff" opacity="0.92"/></svg>
  </div>
  <div>
    <h2 style="margin:0">Fresh Produce Quality — Grading UI</h2>
    <p style="margin:2px 0 0 0; color:#94a3b8;">Upload fruit or vegetable images for automated freshness grading.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# 🔹 LOAD MODEL
# -------------------------------
MODEL_PATH = "freshgrade.keras"
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    'freshapples', 'freshbanana', 'freshbittergroud', 'freshcapsicum', 'freshcucumber',
    'freshokra', 'freshoranges', 'freshpotato', 'freshtomato',
    'rottenapples', 'rottenbanana', 'rottenbittergroud', 'rottencapsicum',
    'rottencucumber', 'rottenokra', 'rottenoranges', 'rottenpotato', 'rottentomato'
]

INPUT_SIZE = (224, 224)

# -------------------------------
# 🔹 IMAGE PREPROCESSING
# -------------------------------
def preprocess_image(img, target_size):
    if hasattr(Image, 'Resampling'):
        resample_method = Image.Resampling.LANCZOS
    else:
        resample_method = Image.ANTIALIAS
    img = ImageOps.fit(img, target_size, resample_method)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# -------------------------------
# 🔹 IMAGE UPLOADER
# -------------------------------
uploaded_files = st.file_uploader(
    "Drag & drop images here or click to browse",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True,
    help="Upload clear close-ups of produce (max 5 images)."
)

if uploaded_files:
    uploaded_files = uploaded_files[:5]
    st.markdown("### Previews")
    cols = st.columns(len(uploaded_files))
    for idx, file in enumerate(uploaded_files):
        image_disp = Image.open(file)
        cols[idx].image(image_disp, use_column_width=True, caption=file.name)

# -------------------------------
# 🔹 ANALYSIS (With A/B/C Grading)
# -------------------------------
if st.button("Analyze Images") and uploaded_files:
    st.markdown("### Analyzing images...")
    progress_bar = st.progress(0)
    for i in range(0, 101, 20):
        time.sleep(0.3)
        progress_bar.progress(i)

    results = []
    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        processed = preprocess_image(img, INPUT_SIZE)
        preds = model.predict(processed)
        class_idx = np.argmax(preds)
        predicted_class = CLASS_NAMES[class_idx]
        confidence = np.max(preds) * 100

        # Grading based on confidence
        if confidence >= 95:
            grade_level = "A"
            grade_color = "🟢"
        elif confidence >= 85:
            grade_level = "B"
            grade_color = "🟡"
        else:
            grade_level = "C"
            grade_color = "🔴"

        # Freshness tag
        if "fresh" in predicted_class.lower():
            grade = f"Fresh ({grade_color} Grade {grade_level})"
            color = "good"
        else:
            grade = f"Spoiled ({grade_color} Grade {grade_level})"
            color = "bad"

        results.append({
            "Filename": file.name,
            "Prediction": predicted_class,
            "Grade": grade,
            "Confidence": f"{confidence:.2f}%",
            "Color": color
        })

    fresh_count = sum(1 for r in results if "Fresh" in r["Grade"])
    overall = (
        "Fresh" if fresh_count >= len(results)/2 else
        "Moderately Fresh" if fresh_count > 0 else
        "Spoiled"
    )
    overall_color = "good" if overall == "Fresh" else "ok" if overall == "Moderately Fresh" else "bad"
    st.markdown(f"### Overall Freshness: <span class='{overall_color}'>{overall}</span>", unsafe_allow_html=True)

    st.markdown("### Detailed Results")
    for r in results:
        st.markdown(
            f"{r['Filename']}: <span class='{r['Color']}'>{r['Grade']}</span> "
            f"(<i>{r['Prediction']}</i> — {r['Confidence']})",
            unsafe_allow_html=True
        )

    df = pd.DataFrame(results)[["Filename", "Prediction", "Grade", "Confidence"]]
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export CSV",
        data=csv,
        file_name="produce_analysis.csv",
        mime="text/csv"
    )

# -------------------------------
# 🔹 INFO PANEL
# -------------------------------
st.markdown("""
<div style="margin-top:20px; padding:14px; border-radius:12px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.03);">
<h4>What this UI does</h4>
<ul style="margin:0; padding-left:18px; color:#94a3b8; font-size:13px; line-height:1.6">
    <li>Upload images (drag & drop or browse).</li>
    <li>Deep learning model predicts Fresh vs Rotten per item.</li>
    <li>Grades assigned by AI confidence: A (≥95%), B (85–94%), C (<85%).</li>
    <li>Download results as CSV.</li>
</ul>
</div>
""", unsafe_allow_html=True)
