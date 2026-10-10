
import cv2
import streamlit as st
import torch
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="AI Face Mask Detection", page_icon="😷")

st.title("AI Face Mask Detection System")
st.write("Upload an image to detect faces and classify mask usage.")

@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="prithivMLmods/Face-Mask-Detection"
    )

@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

model = load_model()
face_detector = load_face_detector()

uploaded_file = st.file_uploader(
    "Upload a photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_cv = cv2.cvtColor(
        __import__("numpy").array(image),
        cv2.COLOR_RGB2BGR
    )

    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.1,
        minNeighbors=5, minSize=(40, 40)
    )

    if len(faces) == 0:
        st.warning("No faces detected. Try another image.")
    else:
        for (x, y, w, h) in faces:
            crop = image.crop((x, y, x + w, y + h))
            predictions = model(crop)
            best = max(predictions, key=lambda p: p["score"])

            label = best["label"].lower()
            score = best["score"]

            if "without" in label or "no_mask" in label or "no mask" in label:
                text = f"No Mask: {score:.0%}"
                color = (0, 0, 255)
            elif "mask" in label:
                text = f"Mask: {score:.0%}"
                color = (0, 180, 0)
            else:
                text = f"{best['label']}: {score:.0%}"
                color = (0, 165, 255)

            cv2.rectangle(
                image_cv, (x, y), (x + w, y + h), color, 2
            )
            cv2.putText(
                image_cv, text, (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        result = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        st.image(result, caption="Detection results",
                 use_container_width=True)
        st.success(f"Faces detected: {len(faces)}") 