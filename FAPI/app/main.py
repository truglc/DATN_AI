from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

import shutil
import os
import cv2

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        boxes = detector.detect(frame)
        tracker.update(boxes)

        pred = violence_model.predict(frame)

        if pred is not None:
            label = "Violence" if pred > 0.5 else "Normal"

            if label == "Violence":
                has_violence = True

            results.append({
                "frame": frame_id,
                "label": label,
                "score": float(pred)
            })

    cap.release()

    # 👉 lưu video vi phạm
    if has_violence:
        shutil.copy(filepath, os.path.join(VIOLENCE_FOLDER, file.filename))

    return {
        "results": results,
        "has_violence": has_violence
    }


@app.get("/history")
def history():
    files = os.listdir(VIOLENCE_FOLDER)
    return {"videos": files}