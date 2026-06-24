# Load model + xử lý AI/core stream

import cv2
import uuid
import time
import numpy as np
from collections import deque, defaultdict

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

from config import CONFIG, MODEL_PATH, SNAPSHOT_DIR, OUTPUT_DIR, cfg
from database import (
    insert_alert, insert_performance, insert_prediction_log, update_video_output
)

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


print("Loading LSTM violence model...")
if MODEL_PATH.exists():
    model_vl = load_model(str(MODEL_PATH))
    print("Loaded model:", MODEL_PATH)
else:
    print("WARNING: Không thấy model:", MODEL_PATH)
    print("Cách sửa: set biến môi trường MODEL_PATH hoặc sửa MODEL_PATH trong app.py")

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

# =========================
# CORE
# =========================
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


def resize_for_display(frame):
    h, w = frame.shape[:2]
    if w <= cfg("MAX_DISPLAY_WIDTH"):
        return frame
    scale = cfg("MAX_DISPLAY_WIDTH") / w
    return cv2.resize(frame, (int(cfg("MAX_DISPLAY_WIDTH")), int(h * scale)))


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
    frame = center_crop_to_ratio(frame_bgr, target_ratio=4 / 3)
    frame = cv2.resize(frame, (int(cfg("CAMERA_CANVAS_WIDTH")), int(cfg("CAMERA_CANVAS_HEIGHT"))))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, gray


def extract_vgg_feature(frame):
    img_size = int(cfg("IMG_SIZE"))
    img = cv2.resize(frame, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img.astype(np.float32), axis=0)
    img = preprocess_input(img)
    return feature_extractor.predict(img, verbose=0)[0].astype(np.float32)


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

        score = max(aspect_score * 0.6 + drop_score * 0.25 + height_drop_score * 0.15, drop_score * 0.7 + aspect_score * 0.3)
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


def draw_tracks(frame, tracks):
    for tr in tracks:
        x1, y1, x2, y2 = tr["bbox"]
        tid = tr["track_id"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.putText(frame, f"ID {tid}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)


def draw_overlay(frame, label, fusion_score, lstm_score, rule_score, person_count,
                 fps, latency_ms, iou, interaction, motion, fall, running):
    if label == "FIGHT":
        color = (0, 0, 255)
    elif str(label).startswith("LOADING"):
        color = (0, 255, 255)
    else:
        color = (0, 255, 0)
    x, y = 12, 28
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.48, 1
    lines = [
        f"{label} | fusion={fusion_score:.2f} | lstm={lstm_score:.2f} | rule={rule_score:.2f}",
        f"persons={person_count} | FPS={fps:.1f} | latency={latency_ms:.0f}ms",
        f"iou={iou:.2f} | inter={interaction:.2f} | motion={motion:.2f} | fall={fall:.2f} | run={running:.2f}",
    ]
    for idx, line in enumerate(lines):
        yy = y + idx * 24
        cv2.putText(frame, line, (x, yy), font, fs, (0, 0, 0), th + 2)
        cv2.putText(frame, line, (x, yy), font, fs, color if idx == 0 else (255, 255, 255), th)


def save_snapshot(frame):
    filename = f"alert_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(SNAPSHOT_DIR / filename), frame)
    return filename


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


def make_video_writer(video_id, frame):
    if not CONFIG["SAVE_OUTPUT_VIDEO"] or video_id is None:
        return None, None
    h, w = frame.shape[:2]
    output_name = f"output_{video_id}_{uuid.uuid4().hex[:8]}.mp4"
    output_path = OUTPUT_DIR / output_name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, 25.0, (w, h))
    if writer.isOpened():
        update_video_output(video_id, output_name)
        return writer, output_name
    return None, None


def process_stream(source, video_id=None, source_name="upload"):
    cap = cv2.VideoCapture(source)
    seq_len = int(cfg("SEQ_LEN"))
    feature_buffer = deque(maxlen=seq_len)
    smooth_buffer = deque(maxlen=int(cfg("SMOOTH_WINDOW")))
    temporal = TemporalFilter()
    track_history = defaultdict(lambda: deque(maxlen=8))

    frame_index = 0
    feature_count = 0
    prev_gray = None
    last_alert_time = 0.0
    last_time = time.time()
    writer = None

    last_tracks = []
    lstm_score = rule_score = fusion_score = 0.0
    iou_score = interaction = motion = fall = running = 0.0
    label = "NO FIGHT"

    while True:
        start = time.time()
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        frame = resize_for_display(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_index == 1:
            writer, _ = make_video_writer(video_id, frame)

        if CONFIG["USE_YOLO"] and frame_index % int(cfg("YOLO_EVERY_N_FRAMES")) == 0:
            detections = detect_people(frame)
            last_tracks = update_tracks_fast(frame, detections)

        update_track_history(track_history, last_tracks)
        person_count = len(last_tracks)
        rule_score, iou_score, interaction, motion, fall, running = compute_rule_score(last_tracks, prev_gray, gray, track_history)

        if frame_index % int(cfg("FEATURE_EVERY_N_FRAMES")) == 0:
            feature_buffer.append(extract_vgg_feature(frame))
            feature_count += 1

        if (model_vl is not None and len(feature_buffer) == seq_len and
            feature_count % int(cfg("LSTM_EVERY_N_FEATURES")) == 0 and
            frame_index % int(cfg("FEATURE_EVERY_N_FRAMES")) == 0):
            lstm_score = model_predict_fight_score(feature_buffer, smooth_buffer)

        if len(feature_buffer) < seq_len:
            fusion_score = 0.0
            label = f"LOADING {len(feature_buffer)}/{seq_len}"
        else:
            fusion_score = cfg("LSTM_WEIGHT") * lstm_score + cfg("RULE_WEIGHT") * rule_score
            fusion_score = float(min(1.0, fusion_score))
            label = temporal.update(fusion_score)

        now = time.time()
        dt = now - last_time
        fps = 1.0 / dt if dt > 0 else 0.0
        latency_ms = (now - start) * 1000.0
        last_time = now

        draw_tracks(frame, last_tracks)
        draw_overlay(frame, label, fusion_score, lstm_score, rule_score, person_count,
                     fps, latency_ms, iou_score, interaction, motion, fall, running)

        if writer is not None:
            writer.write(frame)

        if label == "FIGHT" and now - last_alert_time >= cfg("ALERT_COOLDOWN_SEC"):
            snapshot = save_snapshot(frame)
            insert_alert(video_id, source_name, label, fusion_score, lstm_score, rule_score,
                         frame_index, person_count, fps, latency_ms, snapshot)
            last_alert_time = now

        if frame_index % int(cfg("PERFORMANCE_LOG_EVERY")) == 0:
            insert_performance(video_id, source_name, frame_index, fps, latency_ms, person_count)

        if frame_index % int(cfg("PREDICTION_LOG_EVERY")) == 0:
            insert_prediction_log(video_id, source_name, frame_index, label, fusion_score, lstm_score, rule_score,
                                  iou_score, interaction, motion, fall, running, person_count, fps, latency_ms)

        prev_gray = gray
        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg("JPEG_QUALITY"))])
        if ret:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"

    if writer is not None:
        writer.release()
    cap.release()

