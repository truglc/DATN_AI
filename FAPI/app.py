# app_merged_upload_camera.py
# ============================================================
# Upload Video + Browser Camera AI trong CÙNG 1 Flask app
# ============================================================
# Chức năng:
# 1) Upload video file -> stream kết quả detect trên web
# 2) Browser camera -> JS chụp frame -> POST /predict_frame -> Flask xử lý AI
# 3) Cùng pipeline input: resize/crop -> VGG16 preprocess_input -> fc2 4096 -> LSTM seq 20
# 4) Upload có YOLO/rule; Camera fast mode chỉ dùng VGG16 + LSTM để giảm delay
#
# Chạy local:
#   python app_merged_upload_camera.py
#   mở http://127.0.0.1:5000
#
# Chạy Colab + localtunnel:
#   !python app_merged_upload_camera.py
#   hoặc chạy nền rồi mở https://xxxxx.loca.lt
#
# LƯU Ý MODEL_PATH:
# - Nếu chạy local: đặt model tại ./outputs/best_violence_model.h5
# - Nếu chạy Colab: sửa MODEL_PATH thành đường dẫn Drive của bạn
# ============================================================

import os
import cv2
import uuid
import time
import base64
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque

from flask import (
    Flask, request, redirect, url_for, Response,
    render_template, send_from_directory, jsonify
)

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception as e:
    print("YOLO not available:", e)
    YOLO_AVAILABLE = False

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except Exception as e:
    print("DeepSORT not available:", e)
    DEEPSORT_AVAILABLE = False


# =========================
# PATH CONFIG
# =========================
APP_DIR = Path(__file__).parent
UPLOAD_DIR = APP_DIR / "uploads"
OUTPUT_DIR = APP_DIR / "outputs"
SNAPSHOT_DIR = APP_DIR / "snapshots"
DB_PATH = APP_DIR / "database.db"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
SNAPSHOT_DIR.mkdir(exist_ok=True)

# Ưu tiên biến môi trường MODEL_PATH, không có thì dùng ./outputs/best_violence_model.h5
# Colab ví dụ:
# os.environ["MODEL_PATH"] = "/content/drive/MyDrive/model/best_violence_model.h5"
MODEL_PATH = Path('/content/drive/MyDrive/model/best_violence_model.h5')

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)


# =========================
# CONFIG MODEL / SPEED
# =========================
IMG_SIZE = 224
SEQ_LEN = 20
FEATURE_DIM = 4096

# Upload stream display
MAX_DISPLAY_WIDTH = 480
JPEG_QUALITY = 40

# Browser camera gửi frame về Flask
CAMERA_SEND_FPS = 10
CAMERA_CANVAS_WIDTH = 224
CAMERA_CANVAS_HEIGHT = 224
CAMERA_JPEG_QUALITY = 0.45

# AI chạy theo chu kỳ để đỡ lag
YOLO_EVERY_N_FRAMES = 20
CAMERA_YOLO_EVERY_N_FRAMES = 999999   # camera: tắt YOLO để tăng FPS, chỉ dùng LSTM
FEATURE_EVERY_N_FRAMES = 3        # upload: lấy feature mỗi 3 frame để tăng FPS
CAMERA_FEATURE_EVERY_N_FRAMES = 1 # camera: lấy feature mỗi frame để giảm delay cảnh báo
LSTM_EVERY_N_FEATURES = 1
CAMERA_LSTM_EVERY_N_FEATURES = 1

USE_DEEPSORT = False

# Threshold
LSTM_THRESHOLD = 0.82
FUSION_THRESHOLD = 0.72
SMOOTH_WINDOW = 3
FIGHT_CONFIRM_FRAMES = 3
NOFIGHT_CONFIRM_FRAMES = 2

# Rule
MIN_PERSONS_FOR_INTERACTION = 2
CLOSE_DISTANCE_THRESHOLD = 150
MOTION_THRESHOLD = 28.0
LSTM_WEIGHT = 0.80
RULE_WEIGHT = 0.20

# Logging
ALERT_COOLDOWN_SEC = 5
PERFORMANCE_LOG_EVERY = 60


# =========================
# DATABASE
# =========================
def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            original_name TEXT,
            source TEXT,
            created_at TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            source TEXT,
            label TEXT,
            confidence REAL,
            lstm_score REAL,
            rule_score REAL,
            frame_index INTEGER,
            person_count INTEGER,
            fps REAL,
            latency_ms REAL,
            snapshot TEXT,
            created_at TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            source TEXT,
            frame_index INTEGER,
            fps REAL,
            latency_ms REAL,
            person_count INTEGER,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def insert_video(filename, original_name, source):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO videos(filename, original_name, source, created_at) VALUES (?, ?, ?, ?)",
        (filename, original_name, source, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    video_id = c.lastrowid
    conn.close()
    return video_id


def insert_alert(video_id, source, label, confidence, lstm_score, rule_score,
                 frame_index, person_count, fps, latency_ms, snapshot):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO alerts(video_id, source, label, confidence, lstm_score, rule_score,
                           frame_index, person_count, fps, latency_ms, snapshot, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        video_id, source, label, float(confidence), float(lstm_score), float(rule_score),
        int(frame_index), int(person_count), float(fps), float(latency_ms), snapshot,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()


def insert_performance(video_id, source, frame_index, fps, latency_ms, person_count):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO performance(video_id, source, frame_index, fps, latency_ms, person_count, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        video_id, source, int(frame_index), float(fps), float(latency_ms),
        int(person_count), datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()


init_db()


# =========================
# LOAD MODELS
# =========================
print("Loading LSTM violence model...")
if MODEL_PATH.exists():
    model_vl = load_model(str(MODEL_PATH))
    print("Loaded model:", MODEL_PATH)
else:
    model_vl = None
    print("WARNING: Không thấy model:", MODEL_PATH)
    print("Cách sửa: đặt file tại ./outputs/best_violence_model.h5 hoặc set MODEL_PATH")

print("Loading VGG16 feature extractor...")
base_vgg = VGG16(weights="imagenet", include_top=True)
feature_extractor = Model(inputs=base_vgg.input, outputs=base_vgg.get_layer("fc2").output)

if YOLO_AVAILABLE:
    print("Loading YOLOv8n...")
    model_yolo = YOLO("yolov8n.pt")
else:
    model_yolo = None

if DEEPSORT_AVAILABLE and USE_DEEPSORT:
    print("Loading DeepSORT...")
    tracker = DeepSort(max_age=30)
else:
    tracker = None

print("System loaded.")


# =========================
# CORE CLASS / FUNCTIONS
# =========================
class TemporalFilter:
    def __init__(self):
        self.fight_count = 0
        self.nofight_count = 0
        self.label = "NO FIGHT"

    def update(self, score):
        if score >= FUSION_THRESHOLD:
            self.fight_count += 1
            self.nofight_count = 0
        else:
            self.nofight_count += 1
            self.fight_count = 0

        if self.fight_count >= FIGHT_CONFIRM_FRAMES:
            self.label = "FIGHT"

        if self.nofight_count >= NOFIGHT_CONFIRM_FRAMES:
            self.label = "NO FIGHT"

        return self.label


def resize_for_display(frame):
    h, w = frame.shape[:2]
    if w <= MAX_DISPLAY_WIDTH:
        return frame
    scale = MAX_DISPLAY_WIDTH / w
    return cv2.resize(frame, (MAX_DISPLAY_WIDTH, int(h * scale)))


def center_crop_to_ratio(frame, target_ratio=4 / 3):
    h, w = frame.shape[:2]
    current_ratio = w / h

    if abs(current_ratio - target_ratio) < 0.05:
        return frame

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x1 = (w - new_w) // 2
        return frame[:, x1:x1 + new_w]

    new_h = int(w / target_ratio)
    y1 = (h - new_h) // 2
    return frame[y1:y1 + new_h, :]


def prepare_camera_frame(frame_bgr):
    """
    Camera từ browser -> crop 4:3 -> resize 640x480.
    Sau đó VGG16 vẫn nhận 224x224 trong extract_vgg_feature().
    """
    frame = center_crop_to_ratio(frame_bgr, target_ratio=4 / 3)
    frame = cv2.resize(frame, (CAMERA_CANVAS_WIDTH, CAMERA_CANVAS_HEIGHT))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, gray


def extract_vgg_feature(frame):
    """
    Input chuẩn cho model:
    BGR frame -> resize 224x224 -> RGB -> preprocess_input -> VGG16 fc2 4096
    """
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img.astype(np.float32), axis=0)
    img = preprocess_input(img)
    return feature_extractor.predict(img, verbose=0)[0].astype(np.float32)


def detect_people(frame):
    detections = []
    if model_yolo is None:
        return detections

    results = model_yolo(frame, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id != 0 or conf < 0.35:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        detections.append(([int(x1), int(y1), int(x2 - x1), int(y2 - y1)], conf, "person"))

    return detections


def update_tracks_fast(frame, detections):
    if tracker is None:
        tracks = []
        for i, det in enumerate(detections):
            xywh, conf, cls_name = det
            x, y, w, h = xywh
            tracks.append({"track_id": i + 1, "bbox": (x, y, x + w, y + h)})
        return tracks

    tracks_result = []
    tracks = tracker.update_tracks(detections, frame=frame)

    for tr in tracks:
        if not tr.is_confirmed():
            continue
        l, t, r, b = tr.to_ltrb()
        tracks_result.append({"track_id": tr.track_id, "bbox": (int(l), int(t), int(r), int(b))})

    return tracks_result


def center_of_box(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def compute_interaction_score(tracks):
    if len(tracks) < MIN_PERSONS_FOR_INTERACTION:
        return 0.0

    centers = [center_of_box(t["bbox"]) for t in tracks]
    min_dist = 999999.0

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
            min_dist = min(min_dist, d)

    if min_dist <= CLOSE_DISTANCE_THRESHOLD:
        return 1.0

    return float(max(0.0, 1.0 - (min_dist - CLOSE_DISTANCE_THRESHOLD) / 300.0))


def compute_motion_score(prev_gray, gray):
    if prev_gray is None:
        return 0.0

    diff = cv2.absdiff(prev_gray, gray)
    motion = float(np.mean(diff))

    return 1.0 if motion >= MOTION_THRESHOLD else float(motion / MOTION_THRESHOLD)


def compute_rule_score(tracks, prev_gray, gray):
    interaction = compute_interaction_score(tracks)
    motion = compute_motion_score(prev_gray, gray)
    rule = 0.6 * interaction + 0.4 * motion
    return float(rule), float(interaction), float(motion)


def draw_tracks(frame, tracks):
    for tr in tracks:
        x1, y1, x2, y2 = tr["bbox"]
        tid = tr["track_id"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.putText(frame, f"ID {tid}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)


def draw_overlay(frame, label, fusion_score, lstm_score, rule_score,
                 person_count, fps, latency_ms, interaction, motion):
    """
    Overlay nhỏ, nền trong suốt, không che video.
    Chỉ hiện chữ ở góc trái trên.
    """

    if label == "FIGHT":
        color = (0, 0, 255)      # đỏ
    elif str(label).startswith("LOADING"):
        color = (0, 255, 255)    # vàng
    else:
        color = (0, 255, 0)      # xanh

    x = 12
    y = 28

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness = 1

    line1 = f"{label} | fusion={fusion_score:.2f} | lstm={lstm_score:.2f}"
    line2 = f"persons={person_count} | FPS={fps:.1f}"
    line3 = f"interaction={interaction:.2f} | motion={motion:.2f}"

    # Viền đen mỏng để chữ dễ nhìn, không cần nền đen
    cv2.putText(frame, line1, (x, y), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, line1, (x, y), font, font_scale, color, thickness)

    cv2.putText(frame, line2, (x, y + 24), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, line2, (x, y + 24), font, font_scale, (255, 255, 255), thickness)

    cv2.putText(frame, line3, (x, y + 48), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, line3, (x, y + 48), font, font_scale, (255, 255, 255), thickness)

def save_snapshot(frame):
    filename = f"alert_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(SNAPSHOT_DIR / filename), frame)
    return filename


def model_predict_fight_score(feature_buffer, smooth_buffer):
    """
    Train của bạn đang giả định: pred[0] = fight, pred[1] = nofight.
    Nếu model của bạn ngược nhãn thì đổi dòng score = pred[0] thành pred[1].
    """
    if model_vl is None or len(feature_buffer) < SEQ_LEN:
        return 0.0

    x = np.expand_dims(np.array(feature_buffer, dtype=np.float32), axis=0)
    pred = model_vl.predict(x, verbose=0)[0]

    # Binary sigmoid: pred có thể chỉ có 1 giá trị
    if np.ndim(pred) == 0 or len(np.atleast_1d(pred)) == 1:
        score = float(np.atleast_1d(pred)[0])
    else:
        # Softmax 2 lớp: pred[0] = FIGHT theo code train hiện tại của bạn
        score = float(pred[0])

    smooth_buffer.append(score)
    return float(np.mean(smooth_buffer))


# =========================
# UPLOAD VIDEO PIPELINE
# =========================
def process_stream(source, video_id=None, source_name="upload"):
    cap = cv2.VideoCapture(source)

    feature_buffer = deque(maxlen=SEQ_LEN)
    smooth_buffer = deque(maxlen=SMOOTH_WINDOW)
    temporal = TemporalFilter()

    frame_index = 0
    feature_count = 0

    prev_gray = None
    last_alert_time = 0.0
    last_time = time.time()

    last_tracks = []
    lstm_score = 0.0
    rule_score = 0.0
    fusion_score = 0.0
    interaction = 0.0
    motion = 0.0
    label = "NO FIGHT"

    while True:
        start = time.time()
        ok, frame = cap.read()
        if not ok:
            break

        frame_index += 1
        frame = resize_for_display(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_index % YOLO_EVERY_N_FRAMES == 0:
            detections = detect_people(frame)
            last_tracks = update_tracks_fast(frame, detections)

        person_count = len(last_tracks)
        rule_score, interaction, motion = compute_rule_score(last_tracks, prev_gray, gray)

        if frame_index % FEATURE_EVERY_N_FRAMES == 0:
            feature = extract_vgg_feature(frame)
            feature_buffer.append(feature)
            feature_count += 1

        if (
            model_vl is not None
            and len(feature_buffer) == SEQ_LEN
            and feature_count % LSTM_EVERY_N_FEATURES == 0
            and frame_index % FEATURE_EVERY_N_FRAMES == 0
        ):
            lstm_score = model_predict_fight_score(feature_buffer, smooth_buffer)

        # Chưa đủ 20 feature thì chưa kết luận
        if len(feature_buffer) < SEQ_LEN:
            fusion_score = 0.0
            label = f"LOADING {len(feature_buffer)}/{SEQ_LEN}"
        else:
            # Nếu muốn chặn false positive khi <2 người, bật đoạn này:
            # if person_count < 2:
            #     fusion_score = 0.0
            # else:
            #     fusion_score = LSTM_WEIGHT * lstm_score + RULE_WEIGHT * rule_score
            fusion_score = LSTM_WEIGHT * lstm_score + RULE_WEIGHT * rule_score
            label = temporal.update(fusion_score)

        now = time.time()
        dt = now - last_time
        fps = 1.0 / dt if dt > 0 else 0.0
        latency_ms = (now - start) * 1000.0
        last_time = now

        draw_tracks(frame, last_tracks)
        draw_overlay(frame, label, fusion_score, lstm_score, rule_score,
                     person_count, fps, latency_ms, interaction, motion)

        if label == "FIGHT" and now - last_alert_time >= ALERT_COOLDOWN_SEC:
            snapshot = save_snapshot(frame)
            insert_alert(video_id, source_name, label, fusion_score, lstm_score, rule_score,
                         frame_index, person_count, fps, latency_ms, snapshot)
            last_alert_time = now

        if frame_index % PERFORMANCE_LOG_EVERY == 0:
            insert_performance(video_id, source_name, frame_index, fps, latency_ms, person_count)

        prev_gray = gray

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        ret, buffer = cv2.imencode(".jpg", frame, encode_param)
        if not ret:
            continue

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"

    cap.release()


# =========================
# BROWSER CAMERA STATE
# =========================
camera_feature_buffer = deque(maxlen=SEQ_LEN)
camera_smooth_buffer = deque(maxlen=SMOOTH_WINDOW)
camera_temporal = TemporalFilter()

camera_state = {
    "frame_index": 0,
    "feature_count": 0,
    "prev_gray": None,
    "last_tracks": [],
    "lstm_score": 0.0,
    "rule_score": 0.0,
    "fusion_score": 0.0,
    "last_alert_time": 0.0,
    "last_time": time.time()
}


def reset_camera_state():
    global camera_feature_buffer, camera_smooth_buffer, camera_temporal, camera_state
    camera_feature_buffer.clear()
    camera_smooth_buffer.clear()
    camera_temporal = TemporalFilter()
    camera_state.update({
        "frame_index": 0,
        "feature_count": 0,
        "prev_gray": None,
        "last_tracks": [],
        "lstm_score": 0.0,
        "rule_score": 0.0,
        "fusion_score": 0.0,
        "last_alert_time": 0.0,
        "last_time": time.time()
    })



# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM videos")
    total_videos = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM alerts")
    total_alerts = c.fetchone()[0]
    c.execute("SELECT AVG(fps) FROM performance")
    avg_fps = c.fetchone()[0]
    avg_fps = 0 if avg_fps is None else round(avg_fps, 1)
    conn.close()

    return render_template(
        "index.html",
        total_videos=total_videos,
        total_alerts=total_alerts,
        avg_fps=avg_fps,
        model_path=str(MODEL_PATH),
        seq_len=SEQ_LEN,
        max_width=MAX_DISPLAY_WIDTH,
        yolo_every=YOLO_EVERY_N_FRAMES,
        feature_every=FEATURE_EVERY_N_FRAMES,
        camera_feature_every=CAMERA_FEATURE_EVERY_N_FRAMES,
        use_deepsort=USE_DEEPSORT,
        jpeg_quality=JPEG_QUALITY
    )


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("video")
    if not file:
        return redirect(url_for("index"))

    ext = Path(file.filename).suffix.lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv"]:
        return "Chỉ hỗ trợ .mp4, .avi, .mov, .mkv", 400

    filename = f"{uuid.uuid4().hex}{ext}"
    file.save(UPLOAD_DIR / filename)

    video_id = insert_video(filename, file.filename, "upload")
    return redirect(url_for("video_stream", video_id=video_id))


@app.route("/video_stream/<int:video_id>")
def video_stream(video_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT filename, original_name FROM videos WHERE id=?", (video_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return "Không tìm thấy video", 404

    return render_template(
        "stream.html",
        title=f"Đang nhận diện: {row[1]}",
        description="Luồng này xử lý bằng pipeline upload/video stream.",
        stream_url=url_for("video_feed", video_id=video_id)
    )


@app.route("/video_feed/<int:video_id>")
def video_feed(video_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT filename FROM videos WHERE id=?", (video_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return "Không tìm thấy video", 404

    return Response(
        process_stream(str(UPLOAD_DIR / row[0]), video_id=video_id, source_name="upload"),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/camera_ai")
def camera_ai():
    return render_template(
        "camera.html",
        send_fps=CAMERA_SEND_FPS,
        canvas_w=CAMERA_CANVAS_WIDTH,
        canvas_h=CAMERA_CANVAS_HEIGHT,
        camera_jpeg_quality=CAMERA_JPEG_QUALITY,
        seq_len=SEQ_LEN,
        camera_yolo_every=CAMERA_YOLO_EVERY_N_FRAMES,
        feature_every=CAMERA_FEATURE_EVERY_N_FRAMES,
        lstm_every=CAMERA_LSTM_EVERY_N_FEATURES
    )


@app.route("/reset_camera_ai", methods=["POST"])
def reset_camera_ai():
    reset_camera_state()
    return jsonify({"ok": True})


@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    start_time = time.time()

    try:
        if "image" in request.files:
            img_bytes = request.files["image"].read()
        else:
            data = request.get_json(silent=True)
            if not data or "image" not in data:
                return jsonify({"error": "Thiếu image"}), 400
            image_data = data["image"]
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]
            img_bytes = base64.b64decode(image_data)

        np_arr = np.frombuffer(img_bytes, np.uint8)
        raw_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Lỗi decode frame: {e}"}), 400

    if raw_frame is None:
        return jsonify({"error": "Không đọc được frame"}), 400

    frame, gray = prepare_camera_frame(raw_frame)

    camera_state["frame_index"] += 1
    frame_index = camera_state["frame_index"]

    # CAMERA FAST MODE:
    # Tắt YOLO ở camera để tránh delay. Camera chỉ dùng VGG16 + LSTM.
    # Upload video vẫn giữ YOLO/bounding box như cũ.
    camera_state["last_tracks"] = []
    tracks = []
    person_count = 0

    # Vẫn tính motion nhẹ để hiển thị, nhưng KHÔNG dùng rule_score để kết luận camera.
    rule_score, interaction_score, motion_score = compute_rule_score(
        tracks,
        camera_state["prev_gray"],
        gray
    )

    if frame_index % CAMERA_FEATURE_EVERY_N_FRAMES == 0:
        feature = extract_vgg_feature(frame)
        camera_feature_buffer.append(feature)
        camera_state["feature_count"] += 1

    if (
        model_vl is not None
        and len(camera_feature_buffer) == SEQ_LEN
        and camera_state["feature_count"] % CAMERA_LSTM_EVERY_N_FEATURES == 0
        and frame_index % CAMERA_FEATURE_EVERY_N_FRAMES == 0
    ):
        camera_state["lstm_score"] = model_predict_fight_score(camera_feature_buffer, camera_smooth_buffer)

    lstm_score = float(camera_state["lstm_score"])

    if len(camera_feature_buffer) < SEQ_LEN:
        fusion_score = 0.0
        label = f"LOADING {len(camera_feature_buffer)}/{SEQ_LEN}"
    else:
        # Camera dùng LSTM trực tiếp để giảm delay.
        # Không fusion với YOLO/rule vì YOLO đã tắt cho camera.
        fusion_score = lstm_score
        label = camera_temporal.update(fusion_score)

    camera_state["rule_score"] = float(rule_score)
    camera_state["fusion_score"] = float(fusion_score)
    camera_state["prev_gray"] = gray

    now = time.time()
    dt = now - camera_state["last_time"]
    fps = 1.0 / dt if dt > 0 else 0.0
    camera_state["last_time"] = now

    latency_ms = (time.time() - start_time) * 1000.0

    if label == "FIGHT" and now - camera_state["last_alert_time"] >= ALERT_COOLDOWN_SEC:
        frame_to_save = frame.copy()
        draw_tracks(frame_to_save, tracks)
        draw_overlay(frame_to_save, label, fusion_score, lstm_score, rule_score,
                     person_count, fps, latency_ms, interaction_score, motion_score)
        snapshot = save_snapshot(frame_to_save)
        insert_alert(
            video_id=None,
            source="browser_camera",
            label=label,
            confidence=fusion_score,
            lstm_score=lstm_score,
            rule_score=rule_score,
            frame_index=frame_index,
            person_count=person_count,
            fps=fps,
            latency_ms=latency_ms,
            snapshot=snapshot
        )
        camera_state["last_alert_time"] = now

    if frame_index % PERFORMANCE_LOG_EVERY == 0:
        insert_performance(None, "browser_camera", frame_index, fps, latency_ms, person_count)

    return jsonify({
        "label": label,
        "fusion_score": float(fusion_score),
        "lstm_score": float(lstm_score),
        "rule_score": float(rule_score),
        "interaction_score": float(interaction_score),
        "motion_score": float(motion_score),
        "person_count": int(person_count),
        "fps": float(fps),
        "latency_ms": float(latency_ms),
        "frame_index": int(frame_index),
        "sequence_len": int(len(camera_feature_buffer)),
        "required_sequence": int(SEQ_LEN),
        "feature_count": int(camera_state["feature_count"])
    })


@app.route("/webcam")
def webcam():
    return render_template(
        "stream.html",
        title="Webcam server/local realtime - Colab không phải camera browser",
        description="Luồng này mở webcam trực tiếp từ máy đang chạy server.",
        stream_url=url_for("webcam_feed")
    )


@app.route("/webcam_feed")
def webcam_feed():
    return Response(
        process_stream(0, video_id=None, source_name="server_webcam"),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/videos")
def videos():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, filename, original_name, source, created_at FROM videos ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return render_template("videos.html", videos=rows)


@app.route("/alerts")
def alerts():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT id, video_id, source, label, confidence, lstm_score, rule_score,
               frame_index, person_count, fps, latency_ms, snapshot, created_at
        FROM alerts
        ORDER BY id DESC
        LIMIT 200
    """)
    rows = c.fetchall()
    conn.close()
    return render_template("alerts.html", alerts=rows)


@app.route("/performance")
def performance():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT id, video_id, source, frame_index, fps, latency_ms, person_count, created_at
        FROM performance
        ORDER BY id DESC
        LIMIT 200
    """)
    rows = c.fetchall()
    conn.close()
    return render_template("performance.html", rows=rows)


@app.route("/snapshots/<filename>")
def snapshots(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )
