# video_service.py
# ============================================================
# Video upload stream, browser-camera state/recording, drawing, playback generator.
# ============================================================

import time
import uuid
from collections import deque, defaultdict
from datetime import datetime

import cv2

from config import CONFIG, OUTPUT_DIR, SNAPSHOT_DIR, cfg
import database as db
import ai_engine as ai


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


def make_video_writer(video_id, frame):
    if not CONFIG["SAVE_OUTPUT_VIDEO"] or video_id is None:
        return None, None
    h, w = frame.shape[:2]
    output_name = f"output_{video_id}_{uuid.uuid4().hex[:8]}.mp4"
    output_path = OUTPUT_DIR / output_name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, 25.0, (w, h))
    if writer.isOpened():
        db.update_video_output(video_id, output_name)
        return writer, output_name
    return None, None


def process_stream(source, video_id=None, source_name="upload"):
    cap = cv2.VideoCapture(source)
    seq_len = int(cfg("SEQ_LEN"))
    feature_buffer = deque(maxlen=seq_len)
    smooth_buffer = deque(maxlen=int(cfg("SMOOTH_WINDOW")))
    temporal = ai.TemporalFilter()
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
            detections = ai.detect_people(frame)
            last_tracks = ai.update_tracks_fast(frame, detections)

        ai.update_track_history(track_history, last_tracks)
        person_count = len(last_tracks)
        rule_score, iou_score, interaction, motion, fall, running = ai.compute_rule_score(last_tracks, prev_gray, gray, track_history)

        if frame_index % int(cfg("FEATURE_EVERY_N_FRAMES")) == 0:
            feature_buffer.append(ai.extract_vgg_feature(frame))
            feature_count += 1

        if (ai.model_vl is not None and len(feature_buffer) == seq_len and
            feature_count % int(cfg("LSTM_EVERY_N_FEATURES")) == 0 and
            frame_index % int(cfg("FEATURE_EVERY_N_FRAMES")) == 0):
            lstm_score = ai.model_predict_fight_score(feature_buffer, smooth_buffer)

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
            db.insert_alert(video_id, source_name, label, fusion_score, lstm_score, rule_score,
                            frame_index, person_count, fps, latency_ms, snapshot)
            last_alert_time = now

        if frame_index % int(cfg("PERFORMANCE_LOG_EVERY")) == 0:
            db.insert_performance(video_id, source_name, frame_index, fps, latency_ms, person_count)

        if frame_index % int(cfg("PREDICTION_LOG_EVERY")) == 0:
            db.insert_prediction_log(video_id, source_name, frame_index, label, fusion_score, lstm_score, rule_score,
                                     iou_score, interaction, motion, fall, running, person_count, fps, latency_ms)

        prev_gray = gray
        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg("JPEG_QUALITY"))])
        if ret:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"

    if writer is not None:
        writer.release()
    cap.release()


# =========================
# BROWSER CAMERA STATE
# =========================
camera_feature_buffer = deque(maxlen=int(cfg("SEQ_LEN")))
camera_smooth_buffer = deque(maxlen=int(cfg("SMOOTH_WINDOW")))
camera_temporal = ai.TemporalFilter()
camera_track_history = defaultdict(lambda: deque(maxlen=8))
camera_state = {
    "frame_index": 0,
    "feature_count": 0,
    "prev_gray": None,
    "last_tracks": [],
    "lstm_score": 0.0,
    "rule_score": 0.0,
    "fusion_score": 0.0,
    "last_alert_time": 0.0,
    "last_time": time.time(),
}

camera_record_state = {
    "recording": False,
    "writer": None,
    "output_name": None,
    "output_path": None,
    "video_id": None,
    "start_time": None,
}


def reset_camera_state():
    global camera_feature_buffer, camera_smooth_buffer, camera_temporal, camera_track_history
    camera_feature_buffer = deque(maxlen=int(cfg("SEQ_LEN")))
    camera_smooth_buffer = deque(maxlen=int(cfg("SMOOTH_WINDOW")))
    camera_temporal = ai.TemporalFilter()
    camera_track_history = defaultdict(lambda: deque(maxlen=8))
    camera_state.update({
        "frame_index": 0,
        "feature_count": 0,
        "prev_gray": None,
        "last_tracks": [],
        "lstm_score": 0.0,
        "rule_score": 0.0,
        "fusion_score": 0.0,
        "last_alert_time": 0.0,
        "last_time": time.time(),
    })


def start_camera_recording():
    """Bắt đầu ghi video camera. Gọi ngay khi người dùng bấm Bật camera."""
    stop_camera_recording()

    output_name = f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
    output_path = OUTPUT_DIR / output_name

    w = int(cfg("CAMERA_CANVAS_WIDTH"))
    h = int(cfg("CAMERA_CANVAS_HEIGHT"))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Camera thực tế thường gửi frame chậm hơn send_fps do model xử lý, nên để 5 FPS cho playback gần thực tế hơn.
    writer = cv2.VideoWriter(str(output_path), fourcc, 5.0, (w, h))

    if not writer.isOpened():
        camera_record_state.update({
            "recording": False,
            "writer": None,
            "output_name": None,
            "output_path": None,
            "video_id": None,
            "start_time": None,
        })
        return False, "Không mở được VideoWriter để ghi camera"

    video_id = db.insert_video(output_name, output_name, "browser_camera")

    camera_record_state.update({
        "recording": True,
        "writer": writer,
        "output_name": output_name,
        "output_path": output_path,
        "video_id": video_id,
        "start_time": time.time(),
    })

    return True, output_name


def stop_camera_recording():
    """Dừng ghi camera, release writer và cập nhật output video để xem lại trong /videos."""
    writer = camera_record_state.get("writer")
    output_name = camera_record_state.get("output_name")
    output_path = camera_record_state.get("output_path")
    video_id = camera_record_state.get("video_id")

    if writer is not None:
        writer.release()

    saved = False
    if output_name and output_path and video_id and output_path.exists() and output_path.stat().st_size > 0:
        db.update_video_output(video_id, output_name)
        saved = True

    camera_record_state.update({
        "recording": False,
        "writer": None,
        "output_name": None,
        "output_path": None,
        "video_id": None,
        "start_time": None,
    })

    return saved, output_name, video_id


def write_camera_record_frame(frame, label, fusion_score, lstm_score, rule_score, person_count,
                              fps, latency_ms, iou_score, interaction_score, motion_score,
                              fall_score, running_score, tracks):
    """
    Ghi 1 frame camera vào file video output.
    Mặc định ghi frame đã xử lý kèm bbox + nhãn như bản app.py gốc.
    Nếu muốn lưu video sạch, bỏ 2 dòng draw_tracks/draw_overlay bên dưới.
    """
    if not camera_record_state.get("recording"):
        return
    writer = camera_record_state.get("writer")
    if writer is None:
        return

    record_frame = frame.copy()
    draw_tracks(record_frame, tracks)
    draw_overlay(record_frame, label, fusion_score, lstm_score, rule_score, person_count,
                 fps, latency_ms, iou_score, interaction_score, motion_score, fall_score, running_score)

    w = int(cfg("CAMERA_CANVAS_WIDTH"))
    h = int(cfg("CAMERA_CANVAS_HEIGHT"))
    if record_frame.shape[1] != w or record_frame.shape[0] != h:
        record_frame = cv2.resize(record_frame, (w, h))

    writer.write(record_frame)


def get_output_info(video_id):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute("SELECT original_name, output_filename FROM videos WHERE id=?", (video_id,))
    row = c.fetchone()
    conn.close()
    return row


def output_playback_generator(video_id):
    row = get_output_info(video_id)
    if not row:
        return

    original_name, output_filename = row
    if not output_filename:
        return

    output_path = OUTPUT_DIR / output_filename
    if not output_path.exists() or output_path.stat().st_size <= 0:
        return

    cap = cv2.VideoCapture(str(output_path))
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1 or fps > 60:
        fps = 20.0
    delay = 1.0 / fps

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        time.sleep(delay)

    cap.release()
