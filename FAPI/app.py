from flask import Flask, render_template, request, send_from_directory, redirect, url_for, Response
import os
import sqlite3
from datetime import datetime
import cv2
from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
base_model = VGG16(weights="imagenet")
feature_extractor = Model(
    inputs=base_model.input,
    outputs=base_model.get_layer("fc2").output   # (4096,)
)
SEQ_LEN = 1
THRESHOLD = 0.7

PREDICT_EVERY = 10       # Chỉ dự đoán mỗi 10 frame
JPEG_QUALITY = 65        # Nén JPEG mạnh hơn => stream nhanh hơn
DISPLAY_SIZE = (480, 270)  # Giảm kích thước hiển thị
SPEED_FACTOR = 0.1      # 0.5 = phát nhanh gấp đôi
app = Flask(__name__)
model_vl = load_model(r"E:\data\violence_model.h5")

# ================== CONFIG ==================
UPLOAD_FOLDER = "uploads"
DB = "database.db"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================== DATABASE INIT ==================
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            created_at TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            message TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()

# ================== SAVE VIDEO ==================
def save_video(file):
    original_name = file.filename
    ext = os.path.splitext(original_name)[1].lower()

    if ext != ".mp4":
        raise ValueError("Chỉ hỗ trợ file MP4")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    final_filename = f"{timestamp}_{original_name}"
    final_path = os.path.join(UPLOAD_FOLDER, final_filename)

    file.save(final_path)

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "INSERT INTO videos (filename, created_at) VALUES (?, ?)",
        (final_filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

    return final_filename

# ================== SAVE ALERT ==================
def save_alert(video_id):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "INSERT INTO alerts (video_id, message, created_at) VALUES (?, ?, ?)",
        (
            video_id,
            "Violence detected",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )
    conn.commit()
    conn.close()

# ================== GET DATA ==================
def get_videos():
    conn = sqlite3.connect(DB)
    data = conn.execute(
        "SELECT * FROM videos ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return data


def get_alerts():
    conn = sqlite3.connect(DB)
    data = conn.execute(
        "SELECT * FROM alerts ORDER BY id DESC LIMIT 20"
    ).fetchall()
    conn.close()
    return data


def get_chart_hour():
    conn = sqlite3.connect(DB)
    data = conn.execute("""
        SELECT substr(created_at, 12, 2) as hour, COUNT(*)
        FROM alerts
        GROUP BY hour
        ORDER BY hour
    """).fetchall()
    conn.close()

    labels = [d[0] for d in data]
    values = [d[1] for d in data]

    return labels, values


def get_chart_day():
    conn = sqlite3.connect(DB)
    data = conn.execute("""
        SELECT substr(created_at, 1, 10) as day, COUNT(*)
        FROM alerts
        GROUP BY day
        ORDER BY day DESC
        LIMIT 7
    """).fetchall()
    conn.close()

    labels = [d[0] for d in data][::-1]
    values = [d[1] for d in data][::-1]

    return labels, values



# ================== LIVE CAMERA STREAM ==================
camera = cv2.VideoCapture(0)

def generate_frames():
    global camera

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 360))

        # Hiển thị chữ đơn giản
        cv2.putText(
            frame,
            "LIVE CAMERA",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes +
            b"\r\n"
        )

# ================== ROUTES ==================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist("videos")

        for f in files:
            if f and f.filename != "":
                try:
                    save_video(f)
                except ValueError:
                    print("Bỏ qua file không phải MP4:", f.filename)

        return redirect(url_for("index"))

    labels_hour, values_hour = get_chart_hour()
    labels_day, values_day = get_chart_day()

    videos = get_videos()
    latest_video = videos[0] if videos else None

    return render_template(
        "index.html",
        videos=videos,
        latest_video=latest_video,
        alerts=get_alerts(),
        labels=labels_hour,
        values=values_hour,
        labels_day=labels_day,
        values_day=values_day
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename,
        as_attachment=False
    )


@app.route("/video_feed/<filename>")
def video_feed(filename):
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    if not os.path.exists(video_path):
        return "File not found", 404

    return Response(
        generate_video(video_path),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/fake_alert/<int:video_id>")
def fake_alert(video_id):
    save_alert(video_id)
    return "OK"


@app.route("/live")
def live_page():
    return redirect(url_for("camera_page"))


@app.route("/camera")
def camera_page():
    return render_template("camera.html")


@app.route("/camera_feed")
def camera_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/upload_camera", methods=["POST"])
def upload_camera():
    file = request.files.get("camera_video")

    if file and file.filename != "":
        try:
            save_video(file)
        except ValueError:
            return "Only MP4 allowed", 400

    return "OK"


@app.route("/videos")
def videos_page():
    return render_template("videos.html", videos=get_videos())


@app.route("/alerts")
def alerts_page():
    return render_template("alerts.html", alerts=get_alerts())


@app.route("/history")
def history_page():
    return render_template("history.html", alerts=get_alerts())


@app.route("/settings")
def settings_page():
    return render_template("settings.html")
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def generate_video(video_path):
    import time

    cap = cv2.VideoCapture(video_path)

    # Lấy FPS gốc của video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    delay = (1.0 / fps) * SPEED_FACTOR

    frame_buffer = deque(maxlen=SEQ_LEN)
    frame_count = 0
    last_prob = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Resize để hiển thị
        frame = cv2.resize(frame, DISPLAY_SIZE)

        # Lưu frame vào buffer
        frame_buffer.append(frame.copy())

        # Chỉ dự đoán mỗi 5 frame
        if len(frame_buffer) == SEQ_LEN and frame_count % PREDICT_EVERY == 0:
            try:
                last_prob = predict_violence_sequence(frame_buffer)
            except Exception as e:
                print("Prediction error:", e)

        # Dùng kết quả gần nhất
        prob = last_prob

        # Hiển thị kết quả
        if prob > THRESHOLD:
            color = (0, 0, 255)
            text = f"VIOLENCE {prob:.2f}"
        else:
            color = (0, 255, 0)
            text = f"SAFE {prob:.2f}"

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        # Encode JPEG
        _, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        frame_bytes = buffer.tobytes()

        # Stream frame
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes +
            b"\r\n"
        )

        # Giữ tốc độ gần bằng FPS gốc
        time.sleep(delay)

    cap.release()
def predict_violence_sequence(frames):
    features = []

    for frame in frames:
        img = cv2.resize(frame, (224, 224))
        img = img.astype("float32")
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        # Trích vector 4096 chiều
        feature = feature_extractor(img, training=False).numpy()[0]
        features.append(feature)

    # (20, 4096)
    features = np.array(features)

    # (1, 20, 4096)
    features = np.expand_dims(features, axis=0)

    pred = model_vl.predict(features, verbose=0)[0][0]
    return float(pred)


camera = cv2.VideoCapture(0)

# ================== LIVE CAMERA STREAM ==================
camera = cv2.VideoCapture(0)

def generate_frames():
    global camera

    # Buffer lưu 20 frame để đưa vào mô hình
    frame_buffer = deque(maxlen=SEQ_LEN)

    # Đếm frame để không dự đoán ở mọi frame
    frame_count = 0

    # Kết quả dự đoán gần nhất
    last_prob = 0.0

    # Tránh lưu alert liên tục
    last_alert_time = 0

    import time

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_count += 1

        # Resize để hiển thị
        frame = cv2.resize(frame, DISPLAY_SIZE)

        # Lưu frame vào buffer
        frame_buffer.append(frame.copy())

        # Chỉ chạy mô hình mỗi 5 frame khi đã đủ 20 frame
        if len(frame_buffer) == SEQ_LEN and frame_count % PREDICT_EVERY == 0:
            try:
                last_prob = predict_violence_sequence(frame_buffer)
            except Exception as e:
                print("Camera prediction error:", e)

        prob = last_prob

        # Hiển thị kết quả
        if prob > THRESHOLD:
            color = (0, 0, 255)
            text = f"VIOLENCE {prob:.2f}"

            # Chỉ lưu alert tối đa 1 lần mỗi 10 giây
            if time.time() - last_alert_time > 10:
                try:
                    save_alert(0)   # 0 = camera live, không gắn với video upload
                    last_alert_time = time.time()
                except Exception as e:
                    print("Save alert error:", e)
        else:
            color = (0, 255, 0)
            text = f"SAFE {prob:.2f}"

        # Vẽ text lên frame
        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        # Encode JPEG
        _, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        frame_bytes = buffer.tobytes()

        # Stream frame ra trình duyệt
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes +
            b"\r\n"
        )

# ================== RUN ==================
if __name__ == "__main__":
    app.run(debug=True)