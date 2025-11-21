import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
import requests
import io
import time
import base64
from datetime import datetime

from pymongo import MongoClient
from gridfs import GridFS

# -------------------- MongoDB Connection --------------------
mongo_uri = st.secrets["mongo"]["uri"]

client = MongoClient(mongo_uri)
db = client["microscopy_db"]
collection = db["detections"]
fs = GridFS(db)   # GridFS for storing real image files

# ------------------------------------------------------------

st.set_page_config(layout="wide", page_title="Microscopy ONNX Demo")
st.title("Microscopy Detector (ONNX via Ultralytics)")

MODEL_LOCAL_PATH = "best.onnx"
GDRIVE_FILE_ID = ""
MODEL_IMG_SIZE = 1024

CONF_THRESH = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25)

def download_from_gdrive(file_id, dest):
    if os.path.exists(dest):
        return dest
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise RuntimeError("Download failed")
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1024):
            f.write(chunk)
    return dest

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

def draw_predictions(pil_img, results, conf_thresh=0.25):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()

    counts = {}

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            score = float(box.conf[0])
            cls = int(box.cls[0])

            if score < conf_thresh:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = model.names[cls]

            counts[label] = counts.get(label, 0) + 1

            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            text = f"{label} {score:.2f}"
            tw, th = draw.textsize(text, font=font)
            draw.rectangle([x1, y1-th, x1+tw, y1], fill="red")
            draw.text((x1, y1-th), text, fill="white", font=font)

    return pil_img, counts

# Download model if needed
if GDRIVE_FILE_ID:
    download_from_gdrive(GDRIVE_FILE_ID, MODEL_LOCAL_PATH)

with st.spinner("Loading model..."):
    model = load_model(MODEL_LOCAL_PATH)

st.success("Model loaded successfully.")

uploaded = st.file_uploader("Upload microscope image", type=["png", "jpg", "jpeg"])
camera = st.camera_input("Or take a picture")

if uploaded is None and camera is None:
    st.info("Upload an image to begin.")
else:
    img_bytes = uploaded.read() if uploaded else camera.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    st.image(pil_img, caption="Input Image", width=400)

    if st.button("Run inference"):
        start = time.time()

        results = model.predict(
            source=np.array(pil_img),
            imgsz=MODEL_IMG_SIZE,
            conf=CONF_THRESH,
            verbose=False
        )

        pil_out, counts = draw_predictions(pil_img.copy(), results, CONF_THRESH)

        st.image(pil_out, caption="Detections", use_column_width=True)
        st.write("Counts:", counts)
        st.success(f"Done in {time.time() - start:.2f}s")

        # ---------------------- SAVE TO MONGODB ----------------------

        # Convert output image to bytes
        buffer = io.BytesIO()
        pil_out.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        # Store in GridFS
        file_id = fs.put(image_bytes, filename=f"detect_{datetime.now()}.png")

        # Store metadata in collection
        document = {
            "timestamp": datetime.now(),
            "counts": counts,
            "gridfs_file_id": file_id
        }

        inserted = collection.insert_one(document)

        st.info(f"Saved to MongoDB with document ID: {inserted.inserted_id}")
        st.info(f"Stored image file ID: {file_id}")
