# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import requests
from pymongo import MongoClient, errors
import gridfs
import bcrypt
from datetime import datetime


# -----------------------
# Config
# -----------------------
st.set_page_config(layout="wide", page_title="Microscopy ONNX Demo (Auth + Mongo)")

# Model / DB settings - change if needed
MODEL_LOCAL_PATH = "best.onnx"
GDRIVE_FILE_ID = ""          # optional: set if you want app to download model from Drive
MODEL_IMG_SIZE = 1024
DEFAULT_CONF = 0.25

# -----------------------
# Helpers: Mongo URI, password hashing
# -----------------------
def get_mongo_uri():
    # Prefer Streamlit secrets, fallback to env var
    try:
        mongo_conf = st.secrets.get("mongo")
        if mongo_conf and "uri" in mongo_conf:
            return mongo_conf["uri"]
    except Exception:
        pass
    return os.environ.get("MONGO_URI")

MONGO_URI = get_mongo_uri()
USE_DB = bool(MONGO_URI)

def hash_password(plain_password: str) -> bytes:
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt())

def check_password(plain_password: str, hashed: bytes) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed)
    except Exception:
        return False

# -----------------------
# Download model helper
# -----------------------
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

# -----------------------
# Load model (cached)
# -----------------------
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# -----------------------
# Text size utility (robust across environments)
# -----------------------
def get_text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
        return w, h
    except Exception:
        try:
            return draw.textsize(text, font=font)
        except Exception:
            try:
                return font.getsize(text)
            except Exception:
                return (len(text)*6, 11)

# -----------------------
# Draw detections
# -----------------------
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

# -----------------------
# MongoDB init
# -----------------------
client = None
db = None
fs = None
collection = None
users_col = None
db_error_msg = None
if USE_DB:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()  # triggers connection / auth
        db = client["microscopy_db"]
        fs = gridfs.GridFS(db)
        collection = db["detections"]
        users_col = db["users"]
    except errors.OperationFailure as e:
        db_error_msg = ("MongoDB auth failure. Check username/password and user privileges.")
    except errors.ServerSelectionTimeoutError as e:
        db_error_msg = ("Could not connect to MongoDB Atlas. Check Network Access / IP whitelist.")
    except Exception as e:
        db_error_msg = f"MongoDB connection error: {e}"

# -----------------------
# UI: Auth (Signin / Signup) at top
# -----------------------
st.markdown("<h1 style='text-align:left'>Microscopy Detector (ONNX via Ultralytics + MongoDB)</h1>", unsafe_allow_html=True)
st.write("---")

# Initialize session state
if "user" not in st.session_state:
    st.session_state.user = None  # will store {'username':..., '_id':...}

col_auth_left, col_auth_right = st.columns([1,1])
with col_auth_left:
    st.subheader("Authentication")
    # Show either logged in user or forms
    if st.session_state.user:
        st.success(f"Signed in as: {st.session_state.user.get('username')}")
        if st.button("Logout"):
            st.session_state.user = None
            st.experimental_rerun()
    else:
        # Two buttons that open either sign-in or sign-up form
        auth_choice = st.radio("Choose action", ("Sign In", "Sign Up"))

        if auth_choice == "Sign Up":
            with st.form("signup_form"):
                su_username = st.text_input("Username", key="su_username")
                su_email = st.text_input("Email", key="su_email")
                su_password = st.text_input("Password", type="password", key="su_password")
                su_password2 = st.text_input("Confirm password", type="password", key="su_password2")
                submitted = st.form_submit_button("Create account")
                if submitted:
                    if not USE_DB:
                        st.error("MongoDB URI not configured. Add to Streamlit secrets or env var.")
                    else:
                        if db_error_msg:
                            st.error(db_error_msg)
                        elif not su_username or not su_password:
                            st.error("Provide a username and password.")
                        elif su_password != su_password2:
                            st.error("Passwords do not match.")
                        else:
                            # Check existing user
                            existing = users_col.find_one({"$or": [{"username": su_username}, {"email": su_email}]})
                            if existing:
                                st.error("User with that username or email already exists.")
                            else:
                                hashed = hash_password(su_password)
                                user_doc = {
                                    "username": su_username,
                                    "email": su_email,
                                    "password": hashed,   # bytes
                                    "created_at": datetime.utcnow()
                                }
                                try:
                                    res = users_col.insert_one(user_doc)
                                    st.success("Account created. You can now sign in.")
                                except Exception as e:
                                    st.error(f"Failed to create account: {e}")

        else:  # Sign In
            with st.form("signin_form"):
                si_username = st.text_input("Username or Email", key="si_username")
                si_password = st.text_input("Password", type="password", key="si_password")
                submitted = st.form_submit_button("Sign in")
                if submitted:
                    if not USE_DB:
                        st.error("MongoDB URI not configured. Add to Streamlit secrets or env var.")
                    else:
                        if db_error_msg:
                            st.error(db_error_msg)
                        else:
                            # Find user by username or email
                            user = users_col.find_one({"$or": [{"username": si_username}, {"email": si_username}]})
                            if not user:
                                st.error("User not found.")
                            else:
                                stored_pw = user.get("password")
                                # stored_pw may be bytes or Binary in Mongo; handle both
                                if isinstance(stored_pw, (bytes, bytearray)):
                                    good = check_password(si_password, stored_pw)
                                else:
                                    # if it's a python dict or object, convert
                                    try:
                                        # bson.binary.Binary -> bytes
                                        good = check_password(si_password, bytes(stored_pw))
                                    except Exception:
                                        good = False
                                if good:
                                    st.session_state.user = {"username": user.get("username"), "_id": user.get("_id")}
                                    st.success(f"Signed in: {user.get('username')}")
                                    st.experimental_rerun()
                                else:
                                    st.error("Incorrect password.")

with col_auth_right:
    st.subheader("Model & DB status")
    model_status = "Loaded" if os.path.exists(MODEL_LOCAL_PATH) else "Not found locally"
    st.write(f"Model file: `{MODEL_LOCAL_PATH}` — {model_status}")
    if USE_DB:
        if db_error_msg:
            st.error(f"DB: Error - {db_error_msg}")
        else:
            st.success("DB: Connected")
    else:
        st.info("DB: Not configured. Add MONGO URI in Streamlit secrets or env var.")

st.write("---")

# -----------------------
# Load model; allow auto-download if Drive ID present
# -----------------------
if GDRIVE_FILE_ID:
    try:
        download_from_gdrive(GDRIVE_FILE_ID, MODEL_LOCAL_PATH)
    except Exception as e:
        st.error(f"Failed to download model from Drive: {e}")

with st.spinner("Loading model..."):
    try:
        model = load_model(MODEL_LOCAL_PATH)
        model_names = getattr(model, "names", None)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# -----------------------
# Main interface: detection
# -----------------------
col1, col2 = st.columns([1, 1.2])
with col1:
    st.header("Run Detection")
    conf = st.slider("Confidence threshold", 0.0, 1.0, DEFAULT_CONF)
    uploaded = st.file_uploader("Upload microscope image", type=["png","jpg","jpeg","tif","tiff"])
    camera = st.camera_input("Or take a picture (Chromium browsers)")

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

            # Only save to DB if user signed in
            if not USE_DB:
                st.info("Mongo URI not provided. Skipping DB save.")
            elif db_error_msg:
                st.error(db_error_msg)
            elif not st.session_state.user:
                st.info("Sign in to save this detection to the DB.")
            else:
                try:
                    # Save image bytes to GridFS
                    buf = io.BytesIO()
                    pil_out.save(buf, format="PNG")
                    img_bytes_out = buf.getvalue()
                    file_id = fs.put(img_bytes_out, filename=f"det_{int(time.time())}.png", contentType="image/png")

                    document = {
                        "timestamp": datetime.utcnow(),
                        "counts": counts,
                        "model": MODEL_LOCAL_PATH,
                        "img_gridfs_id": file_id,
                        "user": st.session_state.user.get("username")
                    }
                    insertion_result = collection.insert_one(document)
                    st.success(f"Saved detection to DB. doc_id: {insertion_result.inserted_id}")
                except Exception as e:
                    st.error(f"Failed to save to DB: {e}")

with col2:
    st.header("Instructions / Quick help")
    st.markdown("""
    - Use **Sign Up** to create an account (stored in MongoDB with a hashed password).  
    - Use **Sign In** to sign in and enable saving detections.  
    - If using Streamlit Cloud, add your MongoDB URI in *Manage app → Settings → Secrets* as:
      ```
      [mongo]
      uri = "mongodb+srv://<username>:<password>@cluster0.dkc9xzx.mongodb.net/?retryWrites=true&w=majority"
      ```
    - Or set `MONGO_URI` as an environment variable.
    """)

# end of app.py
