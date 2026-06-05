# config.py
# ============================================================
# Cấu hình hệ thống, đường dẫn thư mục, model path.
# ============================================================

import os
from pathlib import Path

APP_DIR = Path(__file__).parent
UPLOAD_DIR = APP_DIR / "uploads"
OUTPUT_DIR = APP_DIR / "outputs"
SNAPSHOT_DIR = APP_DIR / "snapshots"
DB_PATH = APP_DIR / "database.db"

for folder in [UPLOAD_DIR, OUTPUT_DIR, SNAPSHOT_DIR]:
    folder.mkdir(exist_ok=True)

MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        "/content/drive/MyDrive/model/best_violence_model.h5"
    )
)

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
    "FALL_ASPECT_RATIO_THRESHOLD": 1.15,
    "FALL_CENTER_DROP_THRESHOLD": 25.0,
    "FALL_HEIGHT_DROP_RATIO": 0.25,
    "RUN_SPEED_THRESHOLD": 28.0,

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


def cfg(key):
    return CONFIG[key]
