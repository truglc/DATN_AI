# ai_engine.py
# ============================================================
# AI engine: YOLO, DeepSORT, VGG16 feature extractor, CNN+LSTM, rule-based fusion.
# Fall detection + running abnormal KHÔNG tạo nhãn riêng, chỉ là rule score.
# ============================================================

import time
from collections import deque

import cv2
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

from config import CONFIG, MODEL_PATH, cfg

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

tracker = None
model_vl = None
feature_extractor = None
model_yolo = None


def init_tracker():
    global tracker
    if DEEPSORT_AVAILABLE and CONFIG["USE_DEEPSORT"]:
        tracker = DeepSort(max_age=30)
        print("DeepSORT enabled")
    else:
        tracker = None
        print("DeepSORT disabled or unavailable")


def init_ai_models():
    """Load LSTM violence model, VGG16 feature extractor, YOLO, DeepSORT."""
    global model_vl, feature_extractor, model_yolo

    print("Loading LSTM violence model...")
    if MODEL_PATH.exists():
        model_vl = load_model(str(MODEL_PATH))
        print("Loaded model:", MODEL_PATH)
    else:
        model_vl = None
        print("WARNING: Không thấy model:", MODEL_PATH)
        print("Cách sửa: set biến môi trường MODEL_PATH hoặc sửa MODEL_PATH trong config.py")

    print("Loading VGG16 feature extractor...")
    base_vgg = VGG16(weights="imagenet", include_top=True)
    feature_extractor = Model(inputs=base_vgg.input, outputs=base_vgg.get_layer("fc2").output)

    if YOLO_AVAILABLE:
        print("Loading YOLOv8n...")
        model_yolo = YOLO("yolov8n.pt")
    else:
        model_yolo = None

    init_tracker()
    print("System loaded.")


class TemporalFilter:
    def __init__(self):
        self.fight_count = 0
        self.nofight_count = 0
        self.label = "NO FIGHT"

    def update(self, score):
        if score >= cfg("FUSION_THRESHOLD"):
            self.fight_count += 1
            self.nofight_count = 0
        else:
            self.nofight_count += 1
            self.fight_count = 0

        if self.fight_count >= cfg("FIGHT_CONFIRM_FRAMES"):
            self.label = "FIGHT"
        if self.nofight_count >= cfg("NOFIGHT_CONFIRM_FRAMES"):
            self.label = "NO FIGHT"
        return self.label


def extract_vgg_feature(frame):
    if feature_extractor is None:
        return np.zeros((4096,), dtype=np.float32)
    img_size = int(cfg("IMG_SIZE"))
    img = cv2.resize(frame, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img.astype(np.float32), axis=0)
    img = preprocess_input(img)
    return feature_extractor.predict(img, verbose=0)[0].astype(np.float32)


def model_predict_fight_score(feature_buffer, smooth_buffer):
    if model_vl is None or len(feature_buffer) < cfg("SEQ_LEN"):
        return 0.0
    x = np.expand_dims(np.array(feature_buffer, dtype=np.float32), axis=0)
    pred = model_vl.predict(x, verbose=0)[0]
    if np.ndim(pred) == 0 or len(np.atleast_1d(pred)) == 1:
        score = float(np.atleast_1d(pred)[0])
    else:
        # Quy ước model train: pred[0] = FIGHT, pred[1] = NO FIGHT
        score = float(pred[0])
    smooth_buffer.append(score)
    return float(np.mean(smooth_buffer))


def detect_people(frame):
    detections = []
    if model_yolo is None or not CONFIG["USE_YOLO"]:
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
            tracks.append({"track_id": i + 1, "bbox": (int(x), int(y), int(x + w), int(y + h))})
        return tracks

    tracks_result = []
    ds_tracks = tracker.update_tracks(detections, frame=frame)
    for tr in ds_tracks:
        if not tr.is_confirmed():
            continue
        l, t, r, b = tr.to_ltrb()
        tracks_result.append({"track_id": tr.track_id, "bbox": (int(l), int(t), int(r), int(b))})
    return tracks_result


def center_of_box(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter + 1e-6)


def compute_iou_score(tracks):
    if len(tracks) < 2:
        return 0.0
    max_iou = 0.0
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            max_iou = max(max_iou, box_iou(tracks[i]["bbox"], tracks[j]["bbox"]))
    return float(min(1.0, max_iou / max(0.001, cfg("IOU_THRESHOLD"))))


def compute_interaction_score(tracks):
    if len(tracks) < cfg("MIN_PERSONS_FOR_INTERACTION"):
        return 0.0
    centers = [center_of_box(t["bbox"]) for t in tracks]
    min_dist = 999999.0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
            min_dist = min(min_dist, d)
    close = cfg("CLOSE_DISTANCE_THRESHOLD")
    if min_dist <= close:
        return 1.0
    return float(max(0.0, 1.0 - (min_dist - close) / 300.0))


def compute_motion_score(prev_gray, gray):
    if prev_gray is None:
        return 0.0
    diff = cv2.absdiff(prev_gray, gray)
    motion = float(np.mean(diff))
    return 1.0 if motion >= cfg("MOTION_THRESHOLD") else float(motion / cfg("MOTION_THRESHOLD"))


def update_track_history(track_history, tracks):
    for tr in tracks:
        tid = tr["track_id"]
        box = tr["bbox"]
        cx, cy = center_of_box(box)
        x1, y1, x2, y2 = box
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        track_history[tid].append({"center": (cx, cy), "w": w, "h": h, "bbox": box, "time": time.time()})


def compute_fall_score(track_history, tracks):
    """
    Fall không phải nhãn riêng. Đây là tín hiệu rule:
    - bbox nằm ngang
    - tâm y rơi xuống nhanh
    - chiều cao bbox giảm nhanh
    """
    if not tracks:
        return 0.0
    best = 0.0
    for tr in tracks:
        tid = tr["track_id"]
        hist = track_history.get(tid, [])
        x1, y1, x2, y2 = tr["bbox"]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        aspect = w / h
        aspect_score = min(1.0, aspect / cfg("FALL_ASPECT_RATIO_THRESHOLD")) if aspect >= 0.6 else 0.0

        drop_score = 0.0
        height_drop_score = 0.0
        if len(hist) >= 3:
            old = hist[0]
            now = hist[-1]
            dy = now["center"][1] - old["center"][1]
            drop_score = max(0.0, min(1.0, dy / cfg("FALL_CENTER_DROP_THRESHOLD")))
            height_drop_ratio = max(0.0, (old["h"] - now["h"]) / max(1.0, old["h"]))
            height_drop_score = min(1.0, height_drop_ratio / cfg("FALL_HEIGHT_DROP_RATIO"))

        score = max(
            aspect_score * 0.6 + drop_score * 0.25 + height_drop_score * 0.15,
            drop_score * 0.7 + aspect_score * 0.3
        )
        best = max(best, score)
    return float(min(1.0, best))


def compute_running_score(track_history, tracks):
    """
    Running abnormal không phải nhãn riêng. Đây là tín hiệu rule dựa trên vận tốc track ID.
    """
    best = 0.0
    for tr in tracks:
        tid = tr["track_id"]
        hist = track_history.get(tid, [])
        if len(hist) < 2:
            continue
        p1 = np.array(hist[-2]["center"])
        p2 = np.array(hist[-1]["center"])
        speed = float(np.linalg.norm(p2 - p1))
        best = max(best, min(1.0, speed / cfg("RUN_SPEED_THRESHOLD")))
    return float(best)


def compute_rule_score(tracks, prev_gray, gray, track_history):
    interaction = compute_interaction_score(tracks)
    iou = compute_iou_score(tracks)
    motion = compute_motion_score(prev_gray, gray)
    fall = compute_fall_score(track_history, tracks)
    running = compute_running_score(track_history, tracks)

    # Nếu có té ngã/chạy nhanh thì nó chỉ làm tăng rule_score, nhãn cuối vẫn là FIGHT/NO FIGHT
    rule = (
        cfg("RULE_INTERACTION_WEIGHT") * interaction +
        cfg("RULE_IOU_WEIGHT") * iou +
        cfg("RULE_MOTION_WEIGHT") * motion +
        cfg("RULE_FALL_WEIGHT") * fall +
        cfg("RULE_RUN_WEIGHT") * running
    )
    return float(min(1.0, rule)), float(iou), float(interaction), float(motion), float(fall), float(running)
