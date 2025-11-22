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
import bcrypt

st.set_page_config(layout="wide", page_title="Microscopy ONNX Demo")

# --------------------
# Settings / Constants
# --------------------
MODEL_LOCAL_PATH = "best.onnx"   # change if model path is different (e.g., "models/best.onnx")
GDRIVE_FILE_ID = ""              # optional, fill if you want the app to download the model from Drive
MODEL_IMG_SIZE = 1024
DEFAULT_CONF = 0.25

# --------------------
# Helpers: Mongo URI
# --------------------
def get_mongo_uri():
    # Try Streamlit secrets first, then environment variable MONGO_URI
    try:
        mongo_conf = st.secrets.get("mongo")
        if mongo_conf and "uri" in mongo_conf:
            return mongo_conf["uri"]
    except Exception:
        pass
    return os.environ.get("MONGO_URI")

MONGO_URI = get_mongo_uri()
USE_DB = bool(MONGO_URI)

# --------------------
# Download helper (optional)
# --------------------
def download_from_gdrive(file_id, dest):
    if os.path.exists(dest):
        return dest
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=16384):
            if chunk:
                f.write(chunk)
    return dest

# --------------------
# Load model (cached)
# --------------------
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# --------------------
# Text measurement compatibility helper
# --------------------
def get_text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w, h
    except Exception:
        try:
            return draw.textsize(text, font=font)
        except Exception:
            try:
                return font.getsize(text)
            except Exception:
                return (len(text)*6, 11)

# --------------------
# Drawing + postprocess
# --------------------
def draw_predictions(pil_img, results, conf_thresh=0.25, model_names=None):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    counts = {}
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            try:
                score = float(box.conf[0]) if hasattr(box, "conf") else float(box.confidence)
            except Exception:
                score = float(getattr(box, "confidence", 0.0))
            try:
                cls = int(box.cls[0]) if hasattr(box, "cls") else int(box.cls)
            except Exception:
                cls = int(getattr(box, "class_id", 0))
            if score < conf_thresh:
                continue
            try:
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
            except Exception:
                coords = getattr(box, "xyxy", None)
                if coords is not None:
                    x1, y1, x2, y2 = coords[0].tolist()
                else:
                    continue
            label = (model_names[cls] if model_names and cls < len(model_names) else str(cls))
            counts[label] = counts.get(label, 0) + 1
            draw.rectangle([x1, y1, x2, y2], outline=(255,0,0), width=2)
            text = f"{label} {score:.2f}"
            tw, th = get_text_size(draw, text, font)
            ty1 = max(0, y1 - th)
            draw.rectangle([x1, ty1, x1 + tw, y1], fill=(255,0,0))
            draw.text((x1, ty1), text, fill=(255,255,255), font=font)
    return pil_img, counts

# --------------------
# Auth helpers (bcrypt)
# --------------------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except Exception:
        return False

# --------------------
# Connect DB (if configured)
# --------------------
client = None
db = None
fs = None
users_col = None
detections_col = None
db_error_msg = None

if USE_DB:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()  # test connection & auth
        db = client["microscopy_db"]
        fs = gridfs.GridFS(db)
        users_col = db["users"]
        detections_col = db["detections"]
    except errors.OperationFailure:
        db_error_msg = "MongoDB auth failure: check username/password and privileges."
    except errors.ServerSelectionTimeoutError:
        db_error_msg = ("Could not connect to MongoDB Atlas. Possibly IP not whitelisted. "
                        "Temporarily add 0.0.0.0/0 to Network Access for testing or ensure Streamlit Cloud IPs are allowed.")
    except Exception as e:
        db_error_msg = f"MongoDB connection error: {e}"

# --------------------
# Model download & load
# --------------------
if GDRIVE_FILE_ID:
    try:
        st.info("Downloading model from Google Drive...")
        download_from_gdrive(GDRIVE_FILE_ID, MODEL_LOCAL_PATH)
        st.success("Downloaded model.")
    except Exception as e:
        st.error(f"Downloading model failed: {e}")

with st.spinner("Loading model..."):
    try:
        model = load_model(MODEL_LOCAL_PATH)
        model_names = getattr(model, "names", None)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# --------------------
# Session state: user
# --------------------
if "user" not in st.session_state:
    st.session_state["user"] = None
if "show_signin" not in st.session_state:
    st.session_state["show_signin"] = False
if "show_signup" not in st.session_state:
    st.session_state["show_signup"] = False

# --------------------
# Top UI: title + auth buttons
# --------------------
col_left, col_mid, col_right = st.columns([6, 1, 2])
with col_left:
    st.title("Microscopy Detector (ONNX via Ultralytics + MongoDB)")
with col_right:
    if st.session_state["user"]:
        st.write(f"Signed in: **{st.session_state['user']['username']}**")
        if st.button("Sign out"):
            st.session_state["user"] = None
            st.success("Signed out.")
    else:
        # Two buttons as requested
        if st.button("Sign In"):
            st.session_state["show_signin"] = True
            st.session_state["show_signup"] = False
        if st.button("Sign Up"):
            st.session_state["show_signup"] = True
            st.session_state["show_signin"] = False

# --------------------
# Show Sign In / Sign Up forms when requested
# --------------------
def create_user(username, email, password):
    if not USE_DB:
        return False, "DB not configured."
    # Unique username or email check
    if users_col.find_one({"$or": [{"username": username}, {"email": email}]}):
        return False, "Username or email already exists."
    hashed = hash_password(password)
    doc = {
        "username": username,
        "email": email,
        "password": hashed,
        "created_at": datetime.utcnow()
    }
    users_col.insert_one(doc)
    return True, None

def authenticate_user(user_key, password):
    if not USE_DB:
        return None, "DB not configured."
    user = users_col.find_one({"$or": [{"username": user_key}, {"email": user_key}]})
    if not user:
        return None, "No such user."
    if check_password(password, user["password"]):
        # Do not return password to session
        user_info = {"_id": str(user["_id"]), "username": user["username"], "email": user.get("email")}
        return user_info, None
    return None, "Incorrect password."

# Sign up form
if st.session_state["show_signup"]:
    with st.form("signup_form"):
        st.subheader("Create an account")
        su_username = st.text_input("Username")
        su_email = st.text_input("Email")
        su_password = st.text_input("Password", type="password")
        su_password2 = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Sign Up")
        if submitted:
            if not su_username or not su_email or not su_password:
                st.error("Please fill all fields.")
            elif su_password != su_password2:
                st.error("Passwords do not match.")
            else:
                if not USE_DB:
                    st.error("MongoDB not configured. Cannot create user.")
                elif db_error_msg:
                    st.error(db_error_msg)
                else:
                    ok, msg = create_user(su_username, su_email, su_password)
                    if ok:
                        st.success("Account created. You are signed in.")
                        st.session_state["user"] = {"username": su_username, "email": su_email}
                        st.session_state["show_signup"] = False
                    else:
                        st.error(f"Signup failed: {msg}")

# Sign in form
if st.session_state["show_signin"]:
    with st.form("signin_form"):
        st.subheader("Sign in")
        si_userkey = st.text_input("Username or Email")
        si_password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In")
        if submitted:
            if not si_userkey or not si_password:
                st.error("Fill both fields.")
            else:
                if not USE_DB:
                    st.error("MongoDB not configured. Cannot sign in.")
                elif db_error_msg:
                    st.error(db_error_msg)
                else:
                    user_info, msg = authenticate_user(si_userkey, si_password)
                    if user_info:
                        st.success("Signed in.")
                        st.session_state["user"] = user_info
                        st.session_state["show_signin"] = False
                    else:
                        st.error(f"Sign-in failed: {msg}")

# --------------------
# Main: Detection UI (left) and info (right)
# --------------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.header("Run Detection")
    conf = st.slider("Confidence threshold", 0.0, 1.0, DEFAULT_CONF)
    uploaded = st.file_uploader("Upload microscope image", type=["png","jpg","jpeg","tif","tiff"])
    camera = st.camera_input("Or take a picture (Chromium-based browsers only)")

    if uploaded is None and camera is None:
        st.info("Upload an image or use the camera.")
    else:
        img_bytes = uploaded.read() if uploaded else camera.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        st.image(pil_img, caption="Input image", width=400)

        if st.button("Run inference"):
            start = time.time()
            try:
                results = model.predict(source=np.array(pil_img), imgsz=MODEL_IMG_SIZE, conf=conf, verbose=False)
            except Exception as e:
                st.error(f"Model inference failed: {e}")
                st.stop()

            pil_out, counts = draw_predictions(pil_img.copy(), results, conf_thresh=conf, model_names=model_names)
            st.image(pil_out, caption="Detections", use_column_width=True)
            st.write("Counts:", counts)
            st.success(f"Inference done in {time.time()-start:.2f}s")

            # Save to DB only if user signed in
            if not USE_DB:
                st.info("MongoDB not configured. Skipping DB save.")
            elif db_error_msg:
                st.error(db_error_msg)
            elif not st.session_state["user"]:
                st.warning("Sign in to save this detection to the DB.")
            else:
                try:
                    # Save image to GridFS and document to detections collection
                    buf = io.BytesIO()
                    pil_out.save(buf, format="PNG")
                    img_bytes_out = buf.getvalue()
                    grid_id = fs.put(img_bytes_out, filename=f"det_{int(time.time())}.png", contentType="image/png")
                    document = {
                        "timestamp": datetime.utcnow(),
                        "counts": counts,
                        "model": MODEL_LOCAL_PATH,
                        "img_gridfs_id": grid_id,
                        "img_size": MODEL_IMG_SIZE,
                        "user": st.session_state["user"]["username"]
                    }
                    res = detections_col.insert_one(document)
                    st.success(f"Saved detection to DB. doc_id: {res.inserted_id}")
                except Exception as e:
                    st.error(f"Failed to save detection to DB: {e}")

with col2:
    st.header("Info / Tips")
    st.write("• Sign up or sign in using the buttons on top. You must be signed in to save detections.")
    st.write("• If MongoDB is not configured, you can still run detection locally but saving is disabled.")
    if db_error_msg:
        st.error(db_error_msg)
    if USE_DB and not db_error_msg:
        st.write("Connected to MongoDB.")
        st.write(f"Database: microscopy_db    Users collection: users    Detections collection: detections")
    st.write("")
    st.write("Confidence threshold: set higher to reduce false positives. Typical starting value: 0.25 - 0.5.")
