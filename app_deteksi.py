# app_deteksi.py
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import joblib
import time
from collections import deque, Counter

# ================================================================
# LOAD MODEL
# ================================================================
print("Loading model...")
model  = joblib.load('./model/mlp_model.pkl')
scaler = joblib.load('./model/scaler.pkl')
le     = joblib.load('./model/label_encoder.pkl')
print(f"Model loaded! Kelas: {len(le.classes_)}")

# ================================================================
# KONFIGURASI MODE
# ================================================================
MODES = {
    1: {
        "name"   : "ABJAD",
        "label"  : "Mode: Deteksi Huruf (A-Z)",
        "prefix" : "abjad_",
        "color"  : (0, 255, 120),      # hijau
        "filter" : lambda lbl: lbl.startswith("abjad_")
    },
    2: {
        "name"   : "NUMERIK",
        "label"  : "Mode: Deteksi Angka (0-9)",
        "prefix" : "numerik_",
        "color"  : (0, 200, 255),      # kuning
        "filter" : lambda lbl: lbl.startswith("numerik_")
    },
    3: {
        "name"   : "KATA",
        "label"  : "Mode: Deteksi Kata",
        "prefix" : "kata_",
        "color"  : (255, 100, 200),    # pink
        "filter" : lambda lbl: lbl.startswith("kata_")
    },
}

current_mode = 1   # default: mode abjad

# ================================================================
# MEDIAPIPE SETUP
# ================================================================
base_options = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=mp_vision.RunningMode.IMAGE
)
detector = mp_vision.HandLandmarker.create_from_options(options)


# ================================================================
# FUNGSI HELPER
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


def draw_hand_landmarks_manual(frame, hand_landmarks_list, frame_w, frame_h, color):
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]
    for hand_lm in hand_landmarks_list:
        pts = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in hand_lm]
        for (a, b) in connections:
            cv2.line(frame, pts[a], pts[b], color, 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1)
            cv2.circle(frame, pt, 4, color, 1)


def predict_with_mode(features_normalized, mode_id):
    """
    Prediksi dengan memfilter hanya kelas yang sesuai mode aktif.
    Ini mencegah model menebak kategori yang salah
    (misal: mode abjad tidak akan pernah mengeluarkan hasil kata).
    """
    scaled = scaler.transform([features_normalized])
    proba  = model.predict_proba(scaled)[0]   # probabilitas semua kelas

    mode_filter = MODES[mode_id]["filter"]

    # Buat mask: set probabilitas kelas di luar mode ini menjadi 0
    masked_proba = np.zeros_like(proba)
    for idx, class_label in enumerate(le.classes_):
        if mode_filter(class_label):
            masked_proba[idx] = proba[idx]

    # Jika tidak ada kelas yang lolos filter, return None
    if masked_proba.sum() == 0:
        return None, 0.0

    # Normalisasi ulang probabilitas yang tersisa
    masked_proba = masked_proba / masked_proba.sum()

    best_idx   = np.argmax(masked_proba)
    confidence = float(masked_proba[best_idx])
    label      = le.inverse_transform([best_idx])[0]
    return label, confidence


def format_label(raw_label):
    if raw_label.startswith("abjad_"):
        return raw_label.replace("abjad_", "Huruf: ")
    elif raw_label.startswith("numerik_"):
        return raw_label.replace("numerik_", "Angka: ")
    elif raw_label.startswith("kata_"):
        return raw_label.replace("kata_", "Kata: ")
    return raw_label


def draw_mode_selector(frame, w, current_mode):
    """Gambar tab selector mode di bagian bawah layar."""
    tab_w   = w // 3
    tab_h   = 50
    y_start = frame.shape[0] - tab_h

    for mode_id, mode_info in MODES.items():
        x_start = (mode_id - 1) * tab_w
        x_end   = x_start + tab_w

        # Background tab: aktif lebih terang
        if mode_id == current_mode:
            bg_color = mode_info["color"]
            alpha    = 0.85
        else:
            bg_color = (40, 40, 40)
            alpha    = 0.75

        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start), (x_end, frame.shape[0]), bg_color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Teks tab
        text       = f"[{mode_id}] {mode_info['name']}"
        text_color = (0, 0, 0) if mode_id == current_mode else (180, 180, 180)
        font_scale = 0.65
        thickness  = 2 if mode_id == current_mode else 1

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx = x_start + (tab_w - tw) // 2
        ty = y_start + (tab_h + th) // 2

        cv2.putText(frame, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

        # Garis bawah untuk tab aktif
        if mode_id == current_mode:
            cv2.rectangle(frame, (x_start, frame.shape[0] - 4),
                          (x_end, frame.shape[0]), (255, 255, 255), -1)


# ================================================================
# MAIN LOOP
# ================================================================

def main():
    global current_mode

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    CONFIDENCE_THRESHOLD = 0.60
    BUFFER_SIZE          = 12
    pred_buffer          = deque(maxlen=BUFFER_SIZE)

    current_display = "Arahkan tangan ke kamera"
    current_conf    = 0.0
    fps_time        = time.time()
    prev_mode       = current_mode

    print("=" * 45)
    print("  Kontrol keyboard:")
    print("  [1] Mode Deteksi Huruf  (Abjad A-Z)")
    print("  [2] Mode Deteksi Angka  (0-9)")
    print("  [3] Mode Deteksi Kata   (40 kata)")
    print("  [q] Keluar")
    print("=" * 45)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # ---- Deteksi tangan ----
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result    = detector.detect(mp_image)

        mode_color = MODES[current_mode]["color"]

        if result.hand_landmarks:
            draw_hand_landmarks_manual(frame, result.hand_landmarks, w, h, mode_color)

            features_raw  = landmarks_to_array(result.hand_landmarks)
            features_norm = normalize_two_hands(features_raw)
            label, conf   = predict_with_mode(features_norm, current_mode)

            if label is not None and conf >= CONFIDENCE_THRESHOLD:
                pred_buffer.append(label)

            if pred_buffer:
                stable_label    = Counter(pred_buffer).most_common(1)[0][0]
                current_display = format_label(stable_label)
                current_conf    = conf
        else:
            pred_buffer.clear()
            current_display = "Tidak ada tangan"
            current_conf    = 0.0

        # ---- UI: Panel atas ----
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 115), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Garis warna mode di tepi atas
        cv2.rectangle(frame, (0, 0), (w, 5), mode_color, -1)

        # Label mode aktif (kecil, di atas)
        cv2.putText(frame, MODES[current_mode]["label"], (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 1, cv2.LINE_AA)

        # Hasil prediksi (besar)
        cv2.putText(frame, current_display, (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, mode_color, 3, cv2.LINE_AA)

        # Confidence bar
        if current_conf > 0:
            bar_len   = int(current_conf * 350)
            bar_color = mode_color if current_conf >= 0.75 else \
                        (0, 200, 255) if current_conf >= 0.60 else (80, 80, 255)
            cv2.rectangle(frame, (20, 88), (370, 103), (40, 40, 40), -1)
            cv2.rectangle(frame, (20, 88), (20 + bar_len, 103), bar_color, -1)
            cv2.putText(frame, f"{current_conf*100:.1f}%", (380, 101),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

        # FPS & jumlah tangan
        fps = 1.0 / (time.time() - fps_time + 1e-8)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 140, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        hand_count = len(result.hand_landmarks) if result.hand_landmarks else 0
        cv2.putText(frame, f"Tangan: {hand_count}", (w - 190, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        # ---- UI: Tab mode di bawah ----
        draw_mode_selector(frame, w, current_mode)

        cv2.imshow('Deteksi Bahasa Isyarat BISINDO', frame)

        # ---- Keyboard input ----
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            current_mode = 1
            pred_buffer.clear()
            print("Mode: ABJAD (Huruf A-Z)")
        elif key == ord('2'):
            current_mode = 2
            pred_buffer.clear()
            print("Mode: NUMERIK (Angka 0-9)")
        elif key == ord('3'):
            current_mode = 3
            pred_buffer.clear()
            print("Mode: KATA")

        # Reset buffer saat mode berganti
        if current_mode != prev_mode:
            pred_buffer.clear()
            current_display = "Arahkan tangan ke kamera"
            current_conf    = 0.0
            prev_mode       = current_mode

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()