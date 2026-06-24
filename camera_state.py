
# State + xử lý camera browser


import cv2
import uuid
import time
from datetime import datetime
from collections import deque, defaultdict

from config import OUTPUT_DIR, CONFIG, cfg
from database import (
    insert_video, update_video_output, insert_alert,
    insert_performance, insert_prediction_log
)
from ai_core import (
    TemporalFilter, prepare_camera_frame, detect_people, update_tracks_fast,
    update_track_history, compute_rule_score, extract_vgg_feature,
    model_predict_fight_score, draw_tracks, draw_overlay, save_snapshot, model_yolo
)

camera_feature_buffer = deque(maxlen=int(cfg("SEQ_LEN")))
camera_smooth_buffer = deque(maxlen=int(cfg("SMOOTH_WINDOW")))
camera_temporal = TemporalFilter()
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

# Khi bấm Bật camera: tạo writer để ghi lại video đã xử lý.
# Mỗi frame sau khi AI xử lý xong sẽ được vẽ overlay và ghi vào file này.
camera_record_state = {
    "recording": False,
    "writer": None,
    "output_name": None,
    "output_path": None,
    "video_id": None,
    "start_time": None,
}


def reset_camera_state():
    camera_feature_buffer.clear()
    camera_smooth_buffer.clear()
    camera_track_history.clear()

    camera_temporal.fight_count = 0
    camera_temporal.nofight_count = 0
    camera_temporal.label = "NO FIGHT"

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
    fps = float(cfg("CAMERA_SEND_FPS"))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
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

    video_id = insert_video(output_name, output_name, "browser_camera")

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
        update_video_output(video_id, output_name)
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
    """Ghi 1 frame camera đã có bbox + nhãn vào file video output."""
    if not camera_record_state.get("recording"):
        return
    writer = camera_record_state.get("writer")
    if writer is None:
        return

    record_frame = frame.copy()
    draw_tracks(record_frame, tracks)
    draw_overlay(record_frame, label, fusion_score, lstm_score, rule_score, person_count,
                 fps, latency_ms, iou_score, interaction_score, motion_score, fall_score, running_score)

    # Đảm bảo đúng size của VideoWriter.
    w = int(cfg("CAMERA_CANVAS_WIDTH"))
    h = int(cfg("CAMERA_CANVAS_HEIGHT"))
    if record_frame.shape[1] != w or record_frame.shape[0] != h:
        record_frame = cv2.resize(record_frame, (w, h))

    writer.write(record_frame)
