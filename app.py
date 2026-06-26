from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import joblib
import base64
import cv2
import os
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# ================================================================
# LOAD MODEL
# ================================================================
print("Loading model...")
model  = joblib.load('./model/mlp_model.pkl')
scaler = joblib.load('./model/scaler.pkl')
le     = joblib.load('./model/label_encoder.pkl')
print(f"Model loaded! Total kelas: {len(le.classes_)}")

# ================================================================
# LOAD MEDIAPIPE
# ================================================================
LANDMARKER_PATH = './hand_landmarker.task'
base_options = mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4,
    running_mode=mp_vision.RunningMode.IMAGE
)
detector = mp_vision.HandLandmarker.create_from_options(options)
print("MediaPipe loaded!")

# ================================================================
# HELPER FUNCTIONS
# ================================================================

def landmarks_to_array(hand_landmarks_list):
    features = np.zeros(126)
    for i, hand_lm in enumerate(hand_landmarks_list[:2]):
        lm_array = []
        for lm in hand_lm:
            lm_array.extend([lm.x, lm.y, lm.z])
        features[i*63 : i*63+63] = lm_array
    return features


def normalize_two_hands(features):
    normalized = np.zeros(126)
    for i in range(2):
        hand = features[i*63 : i*63+63]
        if np.all(hand == 0):
            continue
        lm = hand.reshape(21, 3)
        wrist = lm[0].copy()
        lm_norm = lm - wrist
        scale = np.max(np.abs(lm_norm)) + 1e-8
        lm_norm = lm_norm / scale
        normalized[i*63 : i*63+63] = lm_norm.flatten()
    return normalized


def predict_with_mode(features_normalized, mode):
    prefix_map = {
        "abjad"  : "abjad_",
        "numerik": "numerik_",
        "kata"   : "kata_"
    }
    prefix = prefix_map.get(mode, "abjad_")

    scaled = scaler.transform([features_normalized])
    proba  = model.predict_proba(scaled)[0]

    masked_proba = np.zeros_like(proba)
    for idx, class_label in enumerate(le.classes_):
        if class_label.startswith(prefix):
            masked_proba[idx] = proba[idx]

    if masked_proba.sum() == 0:
        return None, 0.0

    masked_proba = masked_proba / masked_proba.sum()
    best_idx     = np.argmax(masked_proba)
    confidence   = float(masked_proba[best_idx])
    label        = le.inverse_transform([best_idx])[0]
    return label, confidence


def decode_base64_image(b64_string):
    """Decode base64 image dari browser menjadi numpy array."""
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    img_bytes = base64.b64decode(b64_string)
    img_pil   = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_np    = np.array(img_pil)
    return img_np


def extract_landmarks_positions(hand_landmarks_list, img_w, img_h):
    """Kirim posisi landmark ke frontend untuk digambar di canvas."""
    all_hands = []
    for hand_lm in hand_landmarks_list[:2]:
        pts = [{"x": lm.x, "y": lm.y} for lm in hand_lm]
        all_hands.append(pts)
    return all_hands


# ================================================================
# ROUTES
# ================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint utama: terima frame base64, return prediksi.
    Body JSON: { "image": "<base64>", "mode": "abjad"|"numerik"|"kata" }
    """
    try:
        data  = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        mode      = data.get('mode', 'abjad')
        b64_image = data['image']

        # Decode image
        img_rgb = decode_base64_image(b64_image)
        h, w    = img_rgb.shape[:2]

        # Deteksi MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result   = detector.detect(mp_image)

        if not result.hand_landmarks:
            return jsonify({
                "detected"   : False,
                "label"      : None,
                "confidence" : 0.0,
                "landmarks"  : [],
                "hand_count" : 0
            })

        # Ekstraksi & prediksi
        features_raw  = landmarks_to_array(result.hand_landmarks)
        features_norm = normalize_two_hands(features_raw)
        label, conf   = predict_with_mode(features_norm, mode)

        # Format label untuk tampilan
        display_map = {
            "abjad_"  : lambda l: l.replace("abjad_", ""),
            "numerik_": lambda l: l.replace("numerik_", ""),
            "kata_"   : lambda l: l.replace("kata_", "")
        }
        display_label = label
        for prefix, fmt in display_map.items():
            if label and label.startswith(prefix):
                display_label = fmt(label)
                break

        # Kirim posisi landmark untuk digambar di canvas
        landmarks_pos = extract_landmarks_positions(result.hand_landmarks, w, h)

        return jsonify({
            "detected"     : True,
            "label"        : display_label,
            "raw_label"    : label,
            "confidence"   : round(conf, 4),
            "landmarks"    : landmarks_pos,
            "hand_count"   : len(result.hand_landmarks)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Endpoint info: daftar semua kelas per kategori."""
    classes = {
        "abjad"  : [c.replace("abjad_", "") for c in le.classes_ if c.startswith("abjad_")],
        "numerik": [c.replace("numerik_", "") for c in le.classes_ if c.startswith("numerik_")],
        "kata"   : [c.replace("kata_", "") for c in le.classes_ if c.startswith("kata_")]
    }
    return jsonify(classes)


@app.route('/api/health', methods=['GET'])
def health():
    """Endpoint cek status server."""
    return jsonify({
        "status" : "ok",
        "model"  : "MLP Classifier",
        "classes": len(le.classes_)
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)