# app.py
# ============================================================
# Flask routes only. Core logic is split into:
# config.py, database.py, ai_engine.py, video_service.py
# ============================================================

import base64
import uuid
from pathlib import Path
import time

import cv2
import numpy as np
from flask import (
    Flask, request, redirect, url_for, Response,
    render_template, send_from_directory, jsonify
)

from config import CONFIG, UPLOAD_DIR, OUTPUT_DIR, SNAPSHOT_DIR, MODEL_PATH, cfg
import database as db
import ai_engine as ai
import video_service as vs

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

# Init database + AI models
# Lưu ý: nếu import app.py trong môi trường test mà không muốn load model ngay,
# có thể chuyển dòng ai.init_ai_models() vào block if __name__ == "__main__".
db.init_db()
ai.init_ai_models()


@app.route("/")
def index():
    total_videos, total_alerts, avg_fps = db.fetch_dashboard_counts()
    return render_template(
        "index.html",
        total_videos=total_videos,
        total_alerts=total_alerts,
        avg_fps=avg_fps,
        model_path=str(MODEL_PATH),
        seq_len=cfg("SEQ_LEN"),
        max_width=cfg("MAX_DISPLAY_WIDTH"),
        yolo_every=cfg("YOLO_EVERY_N_FRAMES"),
        feature_every=cfg("FEATURE_EVERY_N_FRAMES"),
        camera_feature_every=cfg("CAMERA_FEATURE_EVERY_N_FRAMES"),
        use_deepsort=CONFIG["USE_DEEPSORT"],
        jpeg_quality=cfg("JPEG_QUALITY")
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
    video_id = db.insert_video(filename, file.filename, "upload")
    return redirect(url_for("video_stream", video_id=video_id))


@app.route("/video_stream/<int:video_id>")
def video_stream(video_id):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute("SELECT filename, original_name FROM videos WHERE id=?", (video_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return "Không tìm thấy video", 404

    return render_template(
        "stream.html",
        title=f"Đang nhận diện: {row[1]}",
        description="Upload video: YOLO/DeepSORT + CNN/LSTM + rule fusion.",
        stream_url=url_for("video_feed", video_id=video_id)
    )


@app.route("/video_feed/<int:video_id>")
def video_feed(video_id):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute("SELECT filename FROM videos WHERE id=?", (video_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return "Không tìm thấy video", 404

    return Response(
        vs.process_stream(str(UPLOAD_DIR / row[0]), video_id=video_id, source_name="upload"),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/camera_ai")
def camera_ai():
    return render_template(
        "camera.html",
        send_fps=cfg("CAMERA_SEND_FPS"),
        canvas_w=cfg("CAMERA_CANVAS_WIDTH"),
        canvas_h=cfg("CAMERA_CANVAS_HEIGHT"),
        camera_jpeg_quality=cfg("CAMERA_JPEG_QUALITY"),
        seq_len=cfg("SEQ_LEN"),
        camera_yolo_every=cfg("CAMERA_YOLO_EVERY_N_FRAMES"),
        feature_every=cfg("CAMERA_FEATURE_EVERY_N_FRAMES"),
        lstm_every=cfg("CAMERA_LSTM_EVERY_N_FEATURES")
    )


@app.route("/reset_camera_ai", methods=["POST"])
def reset_camera_ai():
    vs.reset_camera_state()
    return jsonify({"ok": True})


@app.route("/start_camera_record", methods=["POST"])
def start_camera_record_route():
    # Bấm Bật camera là reset AI và bắt đầu ghi video luôn.
    vs.reset_camera_state()
    ok, message = vs.start_camera_recording()
    return jsonify({
        "ok": ok,
        "message": message,
        "video_id": vs.camera_record_state.get("video_id"),
        "recording": vs.camera_record_state.get("recording"),
    })


@app.route("/stop_camera_record", methods=["POST"])
def stop_camera_record_route():
    saved, output_name, video_id = vs.stop_camera_recording()
    return jsonify({
        "ok": True,
        "saved": saved,
        "output_video": output_name,
        "video_id": video_id,
        "watch_url": url_for("watch_output", video_id=video_id) if saved and video_id else None
    })


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

    frame, gray = vs.prepare_camera_frame(raw_frame)
    vs.camera_state["frame_index"] += 1
    frame_index = vs.camera_state["frame_index"]

    if CONFIG["CAMERA_USE_YOLO"] and ai.model_yolo is not None and frame_index % int(cfg("CAMERA_YOLO_EVERY_N_FRAMES")) == 0:
        old_use_yolo = CONFIG["USE_YOLO"]
        CONFIG["USE_YOLO"] = True
        detections = ai.detect_people(frame)
        CONFIG["USE_YOLO"] = old_use_yolo
        vs.camera_state["last_tracks"] = ai.update_tracks_fast(frame, detections)

    tracks = vs.camera_state["last_tracks"]
    ai.update_track_history(vs.camera_track_history, tracks)
    person_count = len(tracks)
    rule_score, iou_score, interaction_score, motion_score, fall_score, running_score = ai.compute_rule_score(
        tracks, vs.camera_state["prev_gray"], gray, vs.camera_track_history
    )

    if frame_index % int(cfg("CAMERA_FEATURE_EVERY_N_FRAMES")) == 0:
        vs.camera_feature_buffer.append(ai.extract_vgg_feature(frame))
        vs.camera_state["feature_count"] += 1

    if (ai.model_vl is not None and len(vs.camera_feature_buffer) == cfg("SEQ_LEN") and
        vs.camera_state["feature_count"] % int(cfg("CAMERA_LSTM_EVERY_N_FEATURES")) == 0 and
        frame_index % int(cfg("CAMERA_FEATURE_EVERY_N_FRAMES")) == 0):
        vs.camera_state["lstm_score"] = ai.model_predict_fight_score(vs.camera_feature_buffer, vs.camera_smooth_buffer)

    lstm_score = float(vs.camera_state["lstm_score"])
    if len(vs.camera_feature_buffer) < cfg("SEQ_LEN"):
        fusion_score = 0.0
        label = f"LOADING {len(vs.camera_feature_buffer)}/{cfg('SEQ_LEN')}"
    else:
        fusion_score = cfg("LSTM_WEIGHT") * lstm_score + cfg("RULE_WEIGHT") * rule_score
        fusion_score = float(min(1.0, fusion_score))
        label = vs.camera_temporal.update(fusion_score)

    vs.camera_state["rule_score"] = float(rule_score)
    vs.camera_state["fusion_score"] = float(fusion_score)
    vs.camera_state["prev_gray"] = gray

    now = time.time()
    dt = now - vs.camera_state["last_time"]
    fps = 1.0 / dt if dt > 0 else 0.0
    vs.camera_state["last_time"] = now
    latency_ms = (time.time() - start_time) * 1000.0

    if label == "FIGHT" and now - vs.camera_state["last_alert_time"] >= cfg("ALERT_COOLDOWN_SEC"):
        frame_to_save = frame.copy()
        vs.draw_tracks(frame_to_save, tracks)
        vs.draw_overlay(frame_to_save, label, fusion_score, lstm_score, rule_score, person_count,
                        fps, latency_ms, iou_score, interaction_score, motion_score, fall_score, running_score)
        snapshot = vs.save_snapshot(frame_to_save)
        db.insert_alert(vs.camera_record_state.get("video_id"), "browser_camera", label, fusion_score, lstm_score, rule_score,
                        frame_index, person_count, fps, latency_ms, snapshot)
        vs.camera_state["last_alert_time"] = now

    if frame_index % int(cfg("PERFORMANCE_LOG_EVERY")) == 0:
        db.insert_performance(vs.camera_record_state.get("video_id"), "browser_camera", frame_index, fps, latency_ms, person_count)

    if frame_index % int(cfg("PREDICTION_LOG_EVERY")) == 0:
        db.insert_prediction_log(vs.camera_record_state.get("video_id"), "browser_camera", frame_index, label, fusion_score, lstm_score, rule_score,
                                 iou_score, interaction_score, motion_score, fall_score, running_score,
                                 person_count, fps, latency_ms)

    # Nếu camera đang bật ghi, lưu frame đã xử lý vào output video.
    vs.write_camera_record_frame(frame, label, fusion_score, lstm_score, rule_score, person_count,
                                 fps, latency_ms, iou_score, interaction_score, motion_score,
                                 fall_score, running_score, tracks)

    return jsonify({
        "label": label,
        "fusion_score": float(fusion_score),
        "lstm_score": float(lstm_score),
        "rule_score": float(rule_score),
        "iou_score": float(iou_score),
        "interaction_score": float(interaction_score),
        "motion_score": float(motion_score),
        "fall_score": float(fall_score),
        "running_score": float(running_score),
        "person_count": int(person_count),
        "fps": float(fps),
        "latency_ms": float(latency_ms),
        "frame_index": int(frame_index),
        "sequence_len": int(len(vs.camera_feature_buffer)),
        "required_sequence": int(cfg("SEQ_LEN")),
        "feature_count": int(vs.camera_state["feature_count"])
    })


@app.route("/webcam")
def webcam():
    return render_template(
        "stream.html",
        title="Webcam server/local realtime",
        description="Mở webcam từ máy đang chạy Flask server.",
        stream_url=url_for("webcam_feed")
    )


@app.route("/webcam_feed")
def webcam_feed():
    return Response(
        vs.process_stream(0, video_id=None, source_name="server_webcam"),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/videos")
def videos():
    conn = db.get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT id, filename, original_name, source, created_at, output_filename
        FROM videos
        ORDER BY id DESC
    """)
    rows = c.fetchall()
    conn.close()
    return render_template("videos.html", videos=rows)


@app.route("/alerts")
def alerts():
    conn = db.get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT id, video_id, source, label, confidence, lstm_score, rule_score,
               frame_index, person_count, fps, latency_ms, snapshot, created_at
        FROM alerts ORDER BY id DESC LIMIT 200
    """)
    rows = c.fetchall()
    conn.close()
    return render_template("alerts.html", alerts=rows)


@app.route("/performance")
def performance():
    conn = db.get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT id, video_id, source, frame_index, fps, latency_ms, person_count, created_at
        FROM performance ORDER BY id DESC LIMIT 200
    """)
    rows = c.fetchall()
    conn.close()
    return render_template("performance.html", rows=rows)


@app.route("/prediction_logs")
def prediction_logs():
    conn = db.get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT id, video_id, source, frame_index, label, fusion_score, lstm_score, rule_score,
               iou_score, interaction_score, motion_score, fall_score, running_score,
               person_count, fps, latency_ms, created_at
        FROM prediction_logs ORDER BY id DESC LIMIT 300
    """)
    rows = c.fetchall()
    conn.close()
    return render_template("prediction_logs.html", rows=rows)


@app.route("/config", methods=["GET", "POST"])
def config():
    if request.method == "POST":
        bool_keys = ["USE_YOLO", "USE_DEEPSORT", "CAMERA_USE_YOLO", "SAVE_OUTPUT_VIDEO"]
        for key in bool_keys:
            CONFIG[key] = key in request.form

        for key, old_value in list(CONFIG.items()):
            if key in bool_keys or key not in request.form:
                continue
            raw = request.form.get(key, "").strip()
            try:
                if isinstance(old_value, int) and not isinstance(old_value, bool):
                    CONFIG[key] = int(float(raw))
                elif isinstance(old_value, float):
                    CONFIG[key] = float(raw)
                else:
                    CONFIG[key] = raw
            except ValueError:
                pass

        ai.init_tracker()
        vs.reset_camera_state()
        return redirect(url_for("config"))

    return render_template("config.html", values=CONFIG)


@app.route("/snapshots/<filename>")
def snapshots(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)


@app.route("/outputs/<filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


@app.route("/output_feed/<int:video_id>")
def output_feed(video_id):
    return Response(
        vs.output_playback_generator(video_id),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/watch_output/<int:video_id>")
def watch_output(video_id):
    row = vs.get_output_info(video_id)
    if not row:
        return "Không tìm thấy video", 404

    original_name, output_filename = row
    if not output_filename:
        return "Video này chưa có bản output. Hãy bấm Chạy nhận diện trước.", 404

    output_path = OUTPUT_DIR / output_filename
    if not output_path.exists() or output_path.stat().st_size <= 0:
        return "File output chưa tồn tại hoặc đang rỗng. Hãy chạy nhận diện đến khi video kết thúc hoặc tắt camera để lưu file.", 404

    return render_template(
        "watch_output.html",
        title=f"Xem lại kết quả: {original_name}",
        stream_url=url_for("output_feed", video_id=video_id),
        download_url=url_for("outputs", filename=output_filename),
        output_filename=output_filename
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)
