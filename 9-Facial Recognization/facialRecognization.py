import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import cv2
import time
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Face Verification", layout="wide")
st.title("📷 Real Face Verification & Prediction")
st.write("Enroll with one photo, then capture another photo from camera to verify and identify the person.")

script_dir = os.path.dirname(os.path.abspath(__file__))
default_real_dataset = os.path.join(script_dir, "face Recognization", "Faces")
fallback_dataset = os.path.join(script_dir, "faceDataset")


def get_face_detector():
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    return cv2.CascadeClassifier(cascade_path)


def read_streamlit_image(uploaded):
    if uploaded is None:
        return None
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image


def extract_largest_face(image_bgr, detector):
    if image_bgr is None:
        return None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return image_bgr[y:y + h, x:x + w]


def face_embedding(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, (96, 96))
    emb = gray.astype(np.float32).flatten() / 255.0
    norm = np.linalg.norm(emb) + 1e-10
    return emb / norm


@st.cache_data
def build_reference_db(dataset_root):
    detector = get_face_detector()
    db = {}
    samples_per_person = {}
    total_images = 0

    if not os.path.isdir(dataset_root):
        return db, samples_per_person, total_images

    for person in sorted(os.listdir(dataset_root)):
        person_dir = os.path.join(dataset_root, person)
        if not os.path.isdir(person_dir):
            continue

        embeddings = []
        for file_name in os.listdir(person_dir):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png", ".pgm", ".bmp")):
                continue
            image_path = os.path.join(person_dir, file_name)
            image = cv2.imread(image_path)
            if image is None:
                continue
            face = extract_largest_face(image, detector)
            if face is None:
                continue
            embeddings.append(face_embedding(face))
            total_images += 1

        if embeddings:
            db[person] = np.mean(np.array(embeddings), axis=0)
            samples_per_person[person] = len(embeddings)

    return db, samples_per_person, total_images


def identify_face(query_embedding, reference_db):
    if not reference_db:
        return "Unknown", 0.0

    names = list(reference_db.keys())
    ref_vectors = np.array([reference_db[name] for name in names])
    sims = cosine_similarity(query_embedding.reshape(1, -1), ref_vectors)[0]
    best_idx = int(np.argmax(sims))
    return names[best_idx], float(sims[best_idx])


def save_person_image(image_bgr, person_name, dataset_root):
    person_dir = os.path.join(dataset_root, person_name)
    os.makedirs(person_dir, exist_ok=True)
    file_name = f"{person_name}_{int(time.time() * 1000)}.jpg"
    file_path = os.path.join(person_dir, file_name)
    cv2.imwrite(file_path, image_bgr)
    return file_path


st.header("1) Dataset Setup")
dataset_path = default_real_dataset if os.path.isdir(default_real_dataset) else fallback_dataset
custom_dataset = st.text_input("Dataset folder path", value=dataset_path)
os.makedirs(custom_dataset, exist_ok=True)

detector = get_face_detector()

st.header("2) Add Person Photos (Store in Dataset)")
person_name = st.text_input("Enter person name", placeholder="Example: Akshay Kumar").strip()

register_method = st.radio(
    "Choose how to add images",
    ["Camera Capture", "Upload Images"],
    horizontal=True,
    key="register_method",
)

if register_method == "Camera Capture":
    register_camera = st.camera_input("Capture person photo", key="register_camera")
    if st.button("Save Camera Photo", key="save_camera_photo"):
        if not person_name:
            st.error("Please enter a person name before saving.")
        elif register_camera is None:
            st.error("Please capture a photo first.")
        else:
            register_img = read_streamlit_image(register_camera)
            if register_img is None:
                st.error("Could not read captured image.")
            elif extract_largest_face(register_img, detector) is None:
                st.error("No face detected. Capture a clear front-face image.")
            else:
                saved_path = save_person_image(register_img, person_name, custom_dataset)
                build_reference_db.clear()
                st.success(f"Photo saved: {saved_path}")
                st.rerun()
else:
    register_files = st.file_uploader(
        "Upload one or more photos",
        type=["jpg", "jpeg", "png", "bmp", "pgm"],
        accept_multiple_files=True,
        key="register_uploads",
    )
    if st.button("Save Uploaded Photos", key="save_uploaded_photos"):
        if not person_name:
            st.error("Please enter a person name before saving.")
        elif not register_files:
            st.error("Please upload at least one image.")
        else:
            saved_count = 0
            skipped_count = 0
            for uploaded in register_files:
                image = read_streamlit_image(uploaded)
                if image is None:
                    skipped_count += 1
                    continue
                if extract_largest_face(image, detector) is None:
                    skipped_count += 1
                    continue
                save_person_image(image, person_name, custom_dataset)
                saved_count += 1

            build_reference_db.clear()
            if saved_count > 0:
                st.success(f"Saved {saved_count} image(s) for {person_name}.")
                if skipped_count > 0:
                    st.warning(f"Skipped {skipped_count} image(s) with no detectable face.")
                st.rerun()
            else:
                st.error("No images were saved. Ensure uploaded images contain clear faces.")

with st.spinner("Loading reference faces from dataset..."):
    reference_db, samples_per_person, total_images = build_reference_db(custom_dataset)

col_a, col_b, col_c = st.columns(3)
col_a.metric("People", len(reference_db))
col_b.metric("Face Samples", total_images)
col_c.metric("Dataset Found", "Yes" if os.path.isdir(custom_dataset) else "No")

if not reference_db:
    st.warning("No valid faces found in dataset. Check the folder path and image files.")
else:
    with st.expander("Dataset summary"):
        for name, count in samples_per_person.items():
            st.write(f"- {name}: {count} samples")

st.header("3) Predict New Image")
query_method = st.radio("Choose prediction input", ["Camera Capture", "Upload Photo"], horizontal=True)

if query_method == "Camera Capture":
    query_file = st.camera_input("Capture another photo for prediction")
else:
    query_file = st.file_uploader("Upload query face image", type=["jpg", "jpeg", "png", "bmp"], key="query_upload")

threshold = st.slider("Same-person threshold (cosine similarity)", min_value=0.50, max_value=0.95, value=0.75, step=0.01)

if query_file is not None:
    query_img = read_streamlit_image(query_file)
    if query_img is None:
        st.error("Could not read query image.")
    else:
        query_face = extract_largest_face(query_img, detector)
        if query_face is None:
            st.error("No face detected in query image. Try better lighting and front angle.")
        else:
            st.image(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB), caption="Query Image", use_container_width=True)
            query_emb = face_embedding(query_face)

            st.header("4) Prediction Results")

            predicted_name, dataset_score = identify_face(query_emb, reference_db)
            if dataset_score >= threshold:
                st.success(f"Predicted Person: {predicted_name}")
            else:
                st.warning("Predicted Person: Unknown (low similarity)")
            st.write(f"Best similarity score: {dataset_score:.4f}")