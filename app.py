import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Caltech-101 Object Recognition",
    layout="centered"
)

st.title("🔍 Object Recognition using ResNet-18 (Caltech-101)")
st.write("CPU-only | Image Upload & Webcam Prediction")

# -----------------------------
# Device (CPU ONLY)
# -----------------------------
DEVICE = torch.device("cpu")

# -----------------------------
# Load Model (Cached)
# -----------------------------
@st.cache_resource
def load_model():
    checkpoint = torch.load(
        "resnet18_caltech101_generalized.pth",
        map_location=DEVICE
    )

    class_names = checkpoint["classes"]
    num_classes = len(class_names)
    print(class_names)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()

    return model, class_names


model, class_names = load_model()

# -----------------------------
# Image Transform (MUST match training)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Prediction Function
# -----------------------------
def predict_pil_image(pil_img, threshold=0.5):
    img = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, dim=1)

    confidence = conf.item()

    if confidence < threshold:
        return "Unknown Object", confidence
    else:
        return class_names[idx.item()], confidence


# -----------------------------
# UI Mode Selection
# -----------------------------
mode = st.radio(
    "Choose Input Mode:",
    ["📂 Upload Image", "📷 Webcam"]
)

# =============================
# IMAGE UPLOAD MODE
# =============================
if mode == "📂 Upload Image":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Predict"):
            label, confidence = predict_pil_image(image)

            st.success(f"**Prediction:** {label}")
            st.info(f"**Confidence:** {confidence * 100:.2f}%")

# =============================
# WEBCAM MODE
# =============================
elif mode == "📷 Webcam":

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam not accessible")
            break

        # Convert OpenCV → PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        label, confidence = predict_pil_image(pil_img)

        # Draw label
        cv2.putText(
            frame,
            f"{label} ({confidence*100:.1f}%)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        FRAME_WINDOW.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

    cap.release()

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("📌 Trained on Caltech-101 | ResNet-18 | CPU-Only Deployment")
