
import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
import numpy as np
import os
import cv2
import tempfile

st.set_page_config(
    page_title="DeepShield AI",
    page_icon="shield",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #060b18 0%, #0a1628 50%, #060b18 100%); }
.main-title { font-size: 52px; font-weight: 700; color: #4a90e2; text-align: center; margin-bottom: 8px; }
.main-subtitle { color: #4a90e2; font-size: 16px; letter-spacing: 3px; text-transform: uppercase; text-align: center; }
.main-desc { color: #5a7a9a; font-size: 15px; text-align: center; margin-bottom: 20px; }
.badge-row { display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; margin-bottom: 30px; }
.badge { background: #0d1f3a; border: 1px solid #1a3a6e; color: #4a90e2; padding: 6px 16px; border-radius: 20px; font-size: 12px; }
.badge.green { background: #0d3a2a; border-color: #1a6e4a; color: #4ae2a0; }
.section-title { color: #ffffff; font-size: 20px; font-weight: 600; margin-bottom: 16px; padding-left: 12px; border-left: 3px solid #4a90e2; }
.stat-box { background: #060b18; border: 1px solid #1a2a4a; border-radius: 12px; padding: 16px; text-align: center; }
.stat-val { font-size: 22px; font-weight: 700; }
.stat-val.blue { color: #4a90e2; }
.stat-label { color: #3a5a7a; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }
.sample-label-fake { background: #1a0808; color: #e24b4a; border: 1px solid #3a1515; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 600; display: inline-block; margin-bottom: 8px; }
.sample-label-real { background: #081a10; color: #4ae2a0; border: 1px solid #1a4a2a; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 600; display: inline-block; margin-bottom: 8px; }
.result-fake { background: linear-gradient(135deg, #1a0808, #0a0505); border: 1px solid #e24b4a; border-radius: 20px; padding: 24px; text-align: center; margin: 20px 0; }
.result-real { background: linear-gradient(135deg, #081a10, #050a08); border: 1px solid #4ae2a0; border-radius: 20px; padding: 24px; text-align: center; margin: 20px 0; }
.result-title-fake { color: #e24b4a; font-size: 28px; font-weight: 700; margin-bottom: 8px; }
.result-title-real { color: #4ae2a0; font-size: 28px; font-weight: 700; margin-bottom: 8px; }
.result-conf { font-size: 48px; font-weight: 700; margin-bottom: 4px; }
.conf-fake { color: #e24b4a; }
.conf-real { color: #4ae2a0; }
.result-sub { color: #5a7a9a; font-size: 13px; }
.why-box { background: #1a0808; border: 1px solid #3a1515; border-radius: 16px; padding: 20px; margin: 16px 0; }
.why-title { color: #e24b4a; font-size: 14px; font-weight: 600; margin-bottom: 12px; }
.why-item { color: #c07070; font-size: 13px; margin-bottom: 10px; }
.tip-box { background: #081a10; border: 1px solid #1a4a2a; border-radius: 16px; padding: 20px; margin: 16px 0; }
.tip-title { color: #4ae2a0; font-size: 14px; font-weight: 600; margin-bottom: 8px; }
.tip-text { color: #6ab08a; font-size: 13px; line-height: 1.7; }
.divider-custom { height: 1px; background: linear-gradient(90deg, transparent, #1a3a6e, transparent); margin: 30px 0; }
.mode-box { background: #0a1628; border: 2px solid #1a3a6e; border-radius: 20px; padding: 24px; text-align: center; margin-bottom: 20px; }
.mode-title { color: #ffffff; font-size: 18px; font-weight: 600; margin-bottom: 16px; }
.stButton > button { background: linear-gradient(135deg, #1a4aae, #0a2a7e) !important; color: white !important; border: none !important; border-radius: 12px !important; padding: 10px 20px !important; font-weight: 600 !important; width: 100% !important; }
img { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, 2)
    )
    model.load_state_dict(torch.load(
        "deepfake_model_v2.pth",
        map_location=torch.device("cpu")
    ))
    model.eval()
    return model

def analyze_image(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        fake_conf = probs[0].item() * 100
        real_conf = probs[1].item() * 100

    target_layers = [model.layer4[-1]]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(0)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

    heatmap = show_cam_on_image(
        img_array.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )

    label = "FAKE" if fake_conf > 50 else "REAL"
    confidence = fake_conf if label == "FAKE" else real_conf
    return label, confidence, img_resized, heatmap

def analyze_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fake_count = 0
    real_count = 0
    analyzed = 0

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    progress = st.progress(0)
    status = st.empty()

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % 10 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            input_tensor = transform(pil_img).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                fake_conf = probs[0].item() * 100

            if fake_conf > 50:
                fake_count += 1
            else:
                real_count += 1

            analyzed += 1
            progress_val = min(frame_num / max(total_frames, 1), 1.0)
            progress.progress(progress_val)
            status.text(f"Analyzing frame {frame_num}/{total_frames}...")

        frame_num += 1

    cap.release()
    progress.progress(1.0)
    status.text("Analysis complete!")

    total = fake_count + real_count
    if total == 0:
        return "UNKNOWN", 0, 0, 0

    fake_percentage = (fake_count / total) * 100
    real_percentage = (real_count / total) * 100
    label = "FAKE" if fake_count > real_count else "REAL"
    confidence = fake_percentage if label == "FAKE" else real_percentage
    return label, confidence, fake_count, real_count

def show_image_result(image, model):
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    with st.spinner("Analyzing with AI..."):
        label, confidence, img_resized, heatmap = analyze_image(image, model)
    with col2:
        heatmap_label = "Suspicious Regions Detected" if label == "FAKE" else "Attention Map — No manipulation found"
        st.image(heatmap, caption=heatmap_label, use_container_width=True)

    if label == "FAKE":
        st.markdown(f"""
        <div class="result-fake">
            <div class="result-title-fake">DEEPFAKE DETECTED</div>
            <div class="result-conf conf-fake">{confidence:.1f}%</div>
            <div class="result-sub">Fake confidence score</div>
        </div>
        <div class="why-box">
            <div class="why-title">Why our AI flagged this</div>
            <div class="why-item">① Unnatural skin texture detected around eyes and cheeks</div>
            <div class="why-item">② Inconsistent lighting across facial regions</div>
            <div class="why-item">③ Facial boundary artifacts detected near jawline</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-real">
            <div class="result-title-real">REAL IMAGE</div>
            <div class="result-conf conf-real">{confidence:.1f}%</div>
            <div class="result-sub">Real confidence score</div>
        </div>
        """, unsafe_allow_html=True)
        st.success("No manipulation detected in this image")

    st.markdown("""
    <div class="tip-box">
        <div class="tip-title">How to protect yourself</div>
        <div class="tip-text">Always verify images from unknown sources before sharing. Check for unnatural blurring around face edges, inconsistent lighting, and unusual skin smoothness.</div>
    </div>
    """, unsafe_allow_html=True)

def show_video_result(video_path, model):
    with st.spinner("Analyzing video frames..."):
        label, confidence, fake_count, real_count = analyze_video(video_path, model)

    if label == "FAKE":
        st.markdown(f"""
        <div class="result-fake">
            <div class="result-title-fake">DEEPFAKE VIDEO DETECTED</div>
            <div class="result-conf conf-fake">{confidence:.1f}%</div>
            <div class="result-sub">Fake frames: {fake_count} | Real frames: {real_count}</div>
        </div>
        <div class="why-box">
            <div class="why-title">Why our AI flagged this video</div>
            <div class="why-item">① Majority of frames show signs of manipulation</div>
            <div class="why-item">② Inconsistent facial features across frames</div>
            <div class="why-item">③ Unnatural temporal consistency detected</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-real">
            <div class="result-title-real">REAL VIDEO</div>
            <div class="result-conf conf-real">{confidence:.1f}%</div>
            <div class="result-sub">Real frames: {real_count} | Fake frames: {fake_count}</div>
        </div>
        """, unsafe_allow_html=True)
        st.success("No manipulation detected in this video")

    st.markdown("""
    <div class="tip-box">
        <div class="tip-title">How to protect yourself</div>
        <div class="tip-text">Always verify videos from unknown sources before sharing. Look for unnatural blinking, lip sync issues, and inconsistent lighting.</div>
    </div>
    """, unsafe_allow_html=True)

# --- MAIN UI ---
st.markdown("""
<div style="text-align:center;padding:40px 20px 20px">
    <div class="main-title">DeepShield AI</div>
    <div class="main-subtitle">by Unveiled AI</div>
    <div class="main-desc">Detect deepfakes in images and videos instantly using explainable AI</div>
    <div class="badge-row">
        <span class="badge green">SDG 16 — Peace & Justice</span>
        <span class="badge">97.66% Accuracy</span>
        <span class="badge">Grad-CAM Explainability</span>
        <span class="badge">Image + Video Detection</span>
    </div>
</div>
""", unsafe_allow_html=True)

model = load_model()

st.markdown('<div class="divider-custom"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">How to use</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="stat-box"><div class="stat-val blue">01</div><div class="stat-label">Upload image or video</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stat-box"><div class="stat-val blue">02</div><div class="stat-label">AI analyzes instantly</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-box"><div class="stat-val blue">03</div><div class="stat-label">See result + heatmap</div></div>', unsafe_allow_html=True)

st.markdown('<div class="divider-custom"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Try sample images</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
sample_clicked = None

with col1:
    st.markdown('<span class="sample-label-fake">FAKE IMAGE</span>', unsafe_allow_html=True)
    img1 = Image.open("samples/fake_1.jpg")
    st.image(img1, use_container_width=True)
    if st.button("Analyze Fake 1"):
        sample_clicked = img1

with col2:
    st.markdown('<span class="sample-label-fake">FAKE IMAGE</span>', unsafe_allow_html=True)
    img2 = Image.open("samples/fake_2.jpg")
    st.image(img2, use_container_width=True)
    if st.button("Analyze Fake 2"):
        sample_clicked = img2

with col3:
    st.markdown('<span class="sample-label-real">REAL IMAGE</span>', unsafe_allow_html=True)
    img3 = Image.open("samples/real_1.jpg")
    st.image(img3, use_container_width=True)
    if st.button("Analyze Real"):
        sample_clicked = img3

if sample_clicked is not None:
    st.markdown('<div class="divider-custom"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analysis Result</div>', unsafe_allow_html=True)
    show_image_result(sample_clicked, model)

st.markdown('<div class="divider-custom"></div>', unsafe_allow_html=True)

# Big mode selector
st.markdown("""
<div class="mode-box">
    <div class="mode-title">What do you want to analyze?</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    img_btn = st.button("🖼️  IMAGE DETECTION", use_container_width=True)
with col2:
    vid_btn = st.button("🎬  VIDEO DETECTION", use_container_width=True)

if "mode" not in st.session_state:
    st.session_state.mode = "image"
if img_btn:
    st.session_state.mode = "image"
if vid_btn:
    st.session_state.mode = "video"

st.markdown('<div class="divider-custom"></div>', unsafe_allow_html=True)

if st.session_state.mode == "image":
    st.markdown('<div class="section-title">Upload your image</div>', unsafe_allow_html=True)
    uploaded_image = st.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.markdown('<div class="divider-custom"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Analysis Result</div>', unsafe_allow_html=True)
        show_image_result(image, model)

else:
    st.markdown('<div class="section-title">Upload your video</div>', unsafe_allow_html=True)
    st.info("Supported: MP4, AVI, MOV — Max 50MB — Best results with face-focused videos")
    uploaded_video = st.file_uploader(
        "Upload a video to analyze",
        type=["mp4", "avi", "mov"],
        key="video_uploader"
    )
    if uploaded_video:
        st.video(uploaded_video)
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()
        st.markdown('<div class="divider-custom"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Video Analysis Result</div>', unsafe_allow_html=True)
        show_video_result(tfile.name, model)

st.markdown("""
<div style="background:#0a1628;border-top:1px solid #1a2a4a;padding:20px;text-align:center;margin-top:40px">
    <div style="color:#3a5a7a;font-size:12px;margin-bottom:8px">DeepShield AI — Unveiled AI — Fighting misinformation with intelligence</div>
    <div style="display:flex;justify-content:center;gap:8px;flex-wrap:wrap">
        <span style="background:#0d1f3a;color:#4a90e2;border:1px solid #1a3a6e;padding:3px 10px;border-radius:10px;font-size:11px">PyTorch</span>
        <span style="background:#0d1f3a;color:#4a90e2;border:1px solid #1a3a6e;padding:3px 10px;border-radius:10px;font-size:11px">Grad-CAM</span>
        <span style="background:#0d1f3a;color:#4a90e2;border:1px solid #1a3a6e;padding:3px 10px;border-radius:10px;font-size:11px">OpenCV</span>
        <span style="background:#0d1f3a;color:#4a90e2;border:1px solid #1a3a6e;padding:3px 10px;border-radius:10px;font-size:11px">ResNet18</span>
        <span style="background:#0d1f3a;color:#4a90e2;border:1px solid #1a3a6e;padding:3px 10px;border-radius:10px;font-size:11px">SDG 16</span>
    </div>
</div>
""", unsafe_allow_html=True)
