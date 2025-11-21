# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import requests
import io
import time
import base64
from datetime import datetime

# Mongo
from pymongo import MongoClient, errors

# -------------------------
# Config (edit if needed)
# -------------------------
MODEL_LOCAL_PATH = "best.onnx"   # put best.onnx beside app.py or set GDRIVE_FILE_ID
GDRIVE_FILE_ID = ""              # optional: Google Drive file id to download model at startup
MODEL_IMG_SIZE = 1024            # the image size you exported with
DEFAULT_CONF = 0.25

DB_NAME = "microscopy_db"
COLLECTION_NAME = "detections"

# -------------------------
# Helper: Get Mongo URI
# -------------------------
def get_mongo_uri():
    # 1) try Streamlit secrets
    try:
        return st.secrets["mongo"]["uri"]
    except Exception:
        pass
    # 2) try environment variable fallback
    return os.environ.get("MONGO_URI", None)

# -------------------------
# Download helper (optional)
# -------------------------
def download_from_gdrive(file_id, dest):
    if os.path.exists(dest):
        return dest
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download model from Drive: status {r.status_code}")
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=16384):
            if chunk:
                f.write(chunk)
    return dest

# -------------------------
# Streamlit UI config
# -------------------------
st.set_page_config(layout="wide", page_title="Microscopy ONNX Demo (with MongoDB)")
st.title("Microscopy Detector — save detections to MongoDB Atlas")

# Sidebar controls
conf_slider = st.sidebar.slider("Confidence threshold", 0.0, 1.0, float(DEFAULT_CONF), 0.01)
model_img_size = st.sidebar.number_input("Model input size (imgsz)", value=int(MODEL_IMG_SIZE), step=1)
show_raw_doc = st.sidebar.checkbox("Show stored document preview after save", value=True)

# -------------------------
# Load model (cached)
# -------------------------
@st.cache_resource
def load_model(path):
    model = YOLO(path)
    return model

# download model if GDrive id provided
if GDRIVE_FILE_ID:
    try:
        st.info("Downloading model from Google Drive...")
        download_from_gdrive(GDRIVE_FILE_ID, MODEL_LOCAL_PATH)
        st.success("Downloaded model.")
    except Exception as e:
        st.error(f"Failed to download model: {e}")

# attempt to load model
with st.spinner("Loading model..."):
    try:
        model = load_model(MODEL_LOCAL_PATH)
        st.success("Model loaded.")
    except Exception as e:
        st.error(f"Failed to load model '{MODEL_LOCAL_PATH}': {e}")
        st.stop()

# -------------------------
# Drawing & postprocessing
# -------------------------
def draw_predictions(pil_img, results, conf_thresh=0.25):
    """
    Draw detections on PIL image and return (pil_out, counts_dict).
    Uses ultralytics Results objects.
    """
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    counts = {}
    for r in results:  # r is Results for each image (we pass single image)
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        # boxes: a Boxes object; iterate
        for box in boxes:
            # robust attribute access for different ultralytics versions
            score = float(box.conf[0]) if hasattr(box, "conf") else float(getattr(box, "confidence", 0.0))
            cls = int(box.cls[0]) if hasattr(box, "cls") else int(getattr(box, "cls", 0))
            if score < conf_thresh:
                continue
            xyxy = box.xyxy[0].tolist() if hasattr(box, "xyxy") else None
            if xyxy is None:
                continue
            x1, y1, x2, y2 = xyxy
            label = model.names[cls] if (hasattr(model, "names") and cls < len(model.names)) else str(cls)
            counts[label] = counts.get(label, 0) + 1
            # draw box and label background
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            text = f"{label} {score:.2f}"
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                tw, th = draw.textsize(text, font=font)
            draw.rectangle([x1, y1 - th, x1 + tw, y1], fill=(255, 0, 0))
            draw.text((x1, y1 - th), text, fill=(255, 255, 255), font=font)
    return pil_img, counts

# -------------------------
# MongoDB connection (lazy)
# -------------------------
mongo_uri = get_mongo_uri()
mongo_client = None
mongo_connected = False
if mongo_uri:
    try:
        mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        # quick ping to test connection / auth
        mongo_client.admin.command('ping')
        mongo_connected = True
    except errors.PyMongoError as e:
        st.warning(f"MongoDB connection failed: {e}. You can still run inference but saving will be disabled.")
        mongo_connected = False
else:
    st.info("No MongoDB URI found in Streamlit secrets or MONGO_URI env var. Saving disabled until you set it.")

# prepare DB/collection if connected
if mongo_connected:
    db = mongo_client[DB_NAME]
    collection = db[COLLECTION_NAME]

# -------------------------
# Image upload / camera
# -------------------------
uploaded = st.file_uploader("Upload microscope image", type=["png", "jpg", "jpeg", "tif", "tiff"])
camera = st.camera_input("Or take a picture (Chromium browsers)")

if uploaded is None and camera is None:
    st.info("Upload an image or use the camera to run detection.")
    st.stop()

img_bytes = uploaded.read() if uploaded else camera.read()
orig_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
st.image(orig_pil, caption="Input image", width=420)

# Button: run inference
if st.button("Run inference"):
    start = time.time()
    # predict - pass numpy array
    results = model.predict(source=np.array(orig_pil), imgsz=int(model_img_size), conf=float(conf_slider), verbose=False)
    pil_out, counts = draw_predictions(orig_pil.copy(), results, conf_thresh=float(conf_slider))
    st.image(pil_out, caption="Detections", use_column_width=True)
    st.write("Detected counts:", counts)
    st.success(f"Inference done in {time.time() - start:.2f}s")

    # Convert images to Base64
    def pil_to_base64_str(image: Image.Image, fmt="PNG") -> str:
        buf = io.BytesIO()
        image.save(buf, format=fmt)
        b = base64.b64encode(buf.getvalue()).decode("utf-8")
        return b

    orig_b64 = pil_to_base64_str(orig_pil, fmt="PNG")
    detected_b64 = pil_to_base64_str(pil_out, fmt="PNG")

    # Prepare document
    document = {
        "timestamp": datetime.utcnow(),
        "counts": counts,
        "original_image_base64": orig_b64,
        "detected_image_base64": detected_b64,
        "model": MODEL_LOCAL_PATH,
        "img_size": int(model_img_size)
    }

    # If mongo not connected, inform user and skip saving
    if not mongo_connected:
        st.warning("MongoDB not configured or connection failed. To enable saving: add your Atlas URI to Streamlit secrets or set env var MONGO_URI.")
    else:
        # Show a button to confirm saving (avoid accidental writes)
        if st.button("Save this detection to DB"):
            try:
                insertion_result = collection.insert_one(document)
                st.success(f"Saved to DB. Document id: {insertion_result.inserted_id}")
                if show_raw_doc:
                    st.write("Stored document preview (truncated fields):")
                    preview = document.copy()
                    # Avoid dumping full huge base64 to UI - show trimmed lengths instead
                    preview["original_image_base64"] = f"<base64 {len(orig_b64)} chars>"
                    preview["detected_image_base64"] = f"<base64 {len(detected_b64)} chars>"
                    st.json(preview)
            except errors.OperationFailure as e:
                st.error(f"MongoDB operation failed: {e}")
            except Exception as e:
                st.error(f"Failed to insert document: {e}")
