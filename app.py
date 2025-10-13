import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import time
import torch.nn as nn
import torchvision.models as models
import json

# -------------------------------
# 🔹 PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Fresh Produce Quality Grading", layout="wide")

st.markdown("""
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
""", unsafe_allow_html=True)

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
MODEL_PATH = "freshgrade_new.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 18)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print("✅ Model loaded successfully!")

# -------------------------------
# 🔹 CLASS LABELS
# -------------------------------
classes = [
    'freshapples', 'freshbanana', 'freshbittergroud', 'freshcapsicum', 'freshcucumber',
    'freshokra', 'freshoranges', 'freshpotato', 'freshtomato',
    'rottenapples', 'rottenbanana', 'rottenbittergroud', 'rottencapsicum',
    'rottencucumber', 'rottenokra', 'rottenoranges', 'rottenpotato', 'rottentomato'
]

# -------------------------------
# 🔹 TRANSFORM
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(img):
    img = img.convert("RGB")
    img = transform(img).unsqueeze(0)
    return img.to(device)

# -------------------------------
# 🔹 FILE UPLOAD
# -------------------------------
uploaded_files = st.file_uploader(
    "Drag & drop images here or click to browse",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Upload clear close-up images of produce (max 5)."
)

if uploaded_files:
    uploaded_files = uploaded_files[:5]
    st.markdown("### Previews")
    cols = st.columns(len(uploaded_files))
    for idx, file in enumerate(uploaded_files):
        image_disp = Image.open(file)
        cols[idx].image(image_disp, use_container_width=True, caption=file.name)

# -------------------------------
# 🔹 PREDICTION
# -------------------------------
if st.button("Analyze Images") and uploaded_files:
    st.markdown("### Analyzing images...")
    progress_bar = st.progress(0)
    for i in range(0, 101, 25):
        time.sleep(0.2)
        progress_bar.progress(i)

    results = []
    for file in uploaded_files:
        img = Image.open(file)
        tensor = preprocess_image(img)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, class_idx = torch.max(probs, 1)
            confidence = confidence.item() * 100
            predicted_class = classes[class_idx.item()]

        # -------------------------------
        # 🔹 GRADING LOGIC
        # -------------------------------
        if confidence >= 95:
            grade = "A"; grade_icon = "🟢"
        elif confidence >= 85:
            grade = "B"; grade_icon = "🟡"
        elif confidence >= 70:
            grade = "C"; grade_icon = "🔴"
        else:
            grade = "Uncertain"; grade_icon = "⚪"

        if predicted_class.startswith("fresh"):
            freshness = f"Fresh ({grade_icon} Grade {grade})"
            color = "good"
        elif predicted_class.startswith("rotten"):
            freshness = f"Spoiled ({grade_icon} Grade {grade})"
            color = "bad"
        else:
            freshness = f"Uncertain ({grade_icon})"
            color = "ok"

        # Clean name for display
        item_name = predicted_class.replace("fresh", "").replace("rotten", "").capitalize()
        display_name = ("Fresh " if predicted_class.startswith("fresh") else "Rotten ") + item_name

        results.append({
            "Filename": file.name,
            "Item": display_name,
            "Prediction": predicted_class,
            "Freshness": freshness,
            "Confidence": f"{confidence:.2f}%",
            "Color": color
        })

    # -------------------------------
    # 🔹 DISPLAY RESULTS
    # -------------------------------
    fresh_count = sum(1 for r in results if "Fresh" in r["Freshness"])
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
            f"{r['Filename']}: <span class='{r['Color']}'>{r['Freshness']}</span> "
            f"(<i>{r['Item']}</i> — {r['Confidence']})",
            unsafe_allow_html=True
        )

    df = pd.DataFrame(results)[["Filename", "Item", "Prediction", "Freshness", "Confidence"]]
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📄 Export Results as CSV",
        data=csv,
        file_name="produce_analysis.csv",
        mime="text/csv"
    )

# -------------------------------
# 🔹 INFO PANEL
# -------------------------------
st.markdown("""
<div style="margin-top:20px; padding:14px; border-radius:12px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.03);">
<h4>How this tool works</h4>
<ul style="margin:0; padding-left:18px; color:#94a3b8; font-size:13px; line-height:1.6">
  <li>Detects the type and freshness of produce using a trained MobileNetV2 model.</li>
  <li>Grades freshness confidence:
    <ul>
      <li>🟢 Grade A ≥ 95%</li>
      <li>🟡 Grade B ≥ 85%</li>
      <li>🔴 Grade C ≥ 70%</li>
      <li>⚪ Uncertain &lt; 70%</li>
    </ul>
  </li>
  <li>Exports results as a downloadable CSV report.</li>
</ul>
</div>
""", unsafe_allow_html=True)
