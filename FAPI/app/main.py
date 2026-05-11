from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

import shutil
import os
import cv2
import numpy as np

from app.detector import Detector
from app.tracker import Tracker
from app.violence import ViolenceModel

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

detector = Detector()
tracker = Tracker()
violence_model = ViolenceModel()

UPLOAD_FOLDER = "uploads"
VIOLENCE_FOLDER = "uploads/violence"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIOLENCE_FOLDER, exist_ok=True)


def get_min_pair_distance(track_infos):
    if len(track_infos) < 2:
        return None

    min_distance = None
    for i in range(len(track_infos)):
        for j in range(i + 1, len(track_infos)):
            cx1, cy1 = track_infos[i]["history"][-1]["centroid"]
            cx2, cy2 = track_infos[j]["history"][-1]["centroid"]
            dist = np.hypot(cx1 - cx2, cy1 - cy2)
            if min_distance is None or dist < min_distance:
                min_distance = dist
    return min_distance


def is_possible_fall(history):
    if len(history) < 4:
        return False

    first = history[-4]
    last = history[-1]

    dy = last["centroid"][1] - first["centroid"][1]
    height_first = first["bbox"][3] - first["bbox"][1]
    height_last = last["bbox"][3] - last["bbox"][1]

    if dy > 60 and height_last < height_first * 0.85:
        return True

    return False


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(filepath)

    results = []
    frame_id = 0
    has_violence = False
    alerts = []
    consecutive_violence = 0
    alert_video_types = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        boxes = detector.detect(frame)
        track_infos = tracker.update(boxes)

        pred = violence_model.predict(frame)
        alert_reason = None

        if pred is not None:
            label = "Violence" if pred > 0.5 else "Normal"

            if label == "Violence":
                consecutive_violence += 1
            else:
                consecutive_violence = 0

            close_distance = get_min_pair_distance(track_infos)
            if consecutive_violence >= 2:
                if close_distance is not None and close_distance < 120:
                    alert_reason = "Violence detected between nearby persons"
                else:
                    alert_reason = "Violence detected"

            if alert_reason is not None and "violence" not in alert_video_types:
                alerts.append({
                    "frame": frame_id,
                    "type": "violence",
                    "message": alert_reason,
                    "score": float(pred),
                    "track_count": len(track_infos),
                    "close_distance": float(close_distance) if close_distance is not None else None,
                })
                alert_video_types.add("violence")
                has_violence = True

            for track_info in track_infos:
                if is_possible_fall(track_info["history"]):
                    event_key = ("fall", track_info["track_id"])
                    if event_key not in alert_video_types:
                        alerts.append({
                            "frame": frame_id,
                            "type": "fall",
                            "track_id": track_info["track_id"],
                            "message": "Possible fall or abnormal movement detected",
                        })
                        alert_video_types.add(event_key)

            results.append({
                "frame": frame_id,
                "label": label,
                "score": float(pred),
                "track_count": len(track_infos),
                "alert_reason": alert_reason,
            })

    cap.release()

    # 👉 lưu video vi phạm
    if has_violence:
        shutil.copy(filepath, os.path.join(VIOLENCE_FOLDER, file.filename))

    return {
        "results": results,
        "alerts": alerts,
        "has_violence": has_violence,
        "track_count": len(track_infos),
    }


@app.get("/history")
def history():
    files = os.listdir(VIOLENCE_FOLDER)
    return {"videos": files}