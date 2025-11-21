# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io, os, time, base64
import requests
from pymongo import MongoClient, errors
import gridfs
from datetime import datetime

st.set_page_config(layout="wide", page_title="Microscopy ONNX Demo")

st.title("Microscopy Detector (ONNX via Ultralytics + MongoDB storage)")

# ---------------- Settings ----------------
MODEL_LOCAL_PATH = "best.onnx"   # change if your model is in a subfolder, e.g. "models/best.onnx"
GDRIVE_FILE_ID = ""             # optional: Google Drive file id if model hosted on Drive
MODEL_IMG_SIZE = 1024
DEFAULT_CONF = 0.25
# ------------------------------------------

# Helper to safely get Mongo URI from secrets or env
def get_mongo_uri():
    # First try Streamlit secrets
    try:
        mongo_conf = st.secrets.get("mongo")
        if mongo_conf and "uri" in mongo_conf:
            return mongo_conf["uri"]
    except Exception:
        pass
    # Next, environment variable
    return os.environ.get("MONGO_URI")

MONGO_URI = get_mongo_uri()
USE_DB = bool(MONGO_URI)

# Download helper (small files)
def dow
