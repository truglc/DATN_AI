
# Cấu hình chung, thư mục, Flask app

import os
from pathlib import Path
from flask import Flask

APP_DIR = Path(__file__).parent
UPLOAD_DIR = APP_DIR / "uploads"
OUTPUT_DIR = APP_DIR / "outputs"
SNAPSHOT_DIR = APP_DIR / "snapshots"
DB_PATH = APP_DIR / "database.db"
for d in [UPLOAD_DIR, OUTPUT_DIR, SNAPSHOT_DIR]:
    d.mkdir(exist_ok=True)

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/content/drive/MyDrive/model/best_violence_model.h5"))

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

# =========================
# CONFIG
# =========================
CONFIG = {
    "USE_YOLO": True,
    "USE_DEEPSORT": False,
    "CAMERA_USE_YOLO": False,
    "SAVE_OUTPUT_VIDEO": True,

    "IMG_SIZE": 224,
    "SEQ_LEN": 20,
    "MAX_DISPLAY_WIDTH": 640,
    "JPEG_QUALITY": 55,

    "CAMERA_SEND_FPS": 10,
    "CAMERA_CANVAS_WIDTH": 224,
    "CAMERA_CANVAS_HEIGHT": 224,
    "CAMERA_JPEG_QUALITY": 0.45,

    "YOLO_EVERY_N_FRAMES": 10,
    "CAMERA_YOLO_EVERY_N_FRAMES": 5,
    "FEATURE_EVERY_N_FRAMES": 3,
    "CAMERA_FEATURE_EVERY_N_FRAMES": 1,
    "LSTM_EVERY_N_FEATURES": 1,
    "CAMERA_LSTM_EVERY_N_FEATURES": 1,

    "FUSION_THRESHOLD": 0.72,
    "SMOOTH_WINDOW": 3,
    "FIGHT_CONFIRM_FRAMES": 3,
    "NOFIGHT_CONFIRM_FRAMES": 2,

    "MIN_PERSONS_FOR_INTERACTION": 2,
    "CLOSE_DISTANCE_THRESHOLD": 150.0,
    "MOTION_THRESHOLD": 28.0,
    "IOU_THRESHOLD": 0.05,

    # Fall/running là thành phần rule, không phải nhãn riêng
    "FALL_ASPECT_RATIO_THRESHOLD": 1.15,    # bbox nằm ngang: width/height cao
    "FALL_CENTER_DROP_THRESHOLD": 25.0,     # tâm người rơi xuống nhanh
    "FALL_HEIGHT_DROP_RATIO": 0.25,         # chiều cao bbox giảm nhanh
    "RUN_SPEED_THRESHOLD": 28.0,            # pixels/frame theo track ID

    # Fusion weights
    "LSTM_WEIGHT": 0.70,
    "RULE_WEIGHT": 0.30,
    "RULE_INTERACTION_WEIGHT": 0.25,
    "RULE_IOU_WEIGHT": 0.20,
    "RULE_MOTION_WEIGHT": 0.20,
    "RULE_FALL_WEIGHT": 0.20,
    "RULE_RUN_WEIGHT": 0.15,

    "ALERT_COOLDOWN_SEC": 5,
    "PERFORMANCE_LOG_EVERY": 60,
    "PREDICTION_LOG_EVERY": 10,
}

tracker = None
model_vl = None
feature_extractor = None

tracker = None
model_vl = None
feature_extractor = None
model_yolo = None


def cfg(key):
    return CONFIG[key]
