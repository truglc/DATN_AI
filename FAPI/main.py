import os
import uuid
import shutil

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from tracker import process_video

app = FastAPI()

# =========================
# CONFIG
# =========================
templates = Jinja2Templates(directory="templates")

INPUT_DIR = "videos/input"
OUTPUT_DIR = "videos/output"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# WEB UI
# =========================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# =========================
# UPLOAD
# =========================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    input_path = os.path.join(INPUT_DIR, f"{video_id}.mp4")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"video_id": video_id}

# =========================
# PROCESS
# =========================
@app.post("/process/{video_id}")
def process(video_id: str):
    input_path = os.path.join(INPUT_DIR, f"{video_id}.mp4")
    output_path = os.path.join(OUTPUT_DIR, f"{video_id}.mp4")

    if not os.path.exists(input_path):
        return {"error": "Video không tồn tại"}

    process_video(input_path, output_path)

    return {"result_url": f"/result/{video_id}"}

# =========================
# RESULT
# =========================
@app.get("/result/{video_id}")
def result(video_id: str):
    path = os.path.join(OUTPUT_DIR, f"{video_id}.mp4")

    if not os.path.exists(path):
        return {"error": "Chưa có kết quả"}

    return FileResponse(path, media_type="video/mp4")