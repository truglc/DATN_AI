const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const labelBox = document.getElementById("label");
const scoresBox = document.getElementById("scores");
const infoBox = document.getElementById("info");
const seqBox = document.getElementById("seq");
const statusBox = document.getElementById("status");

let stream = null;
let running = false;
let sending = false;
let timer = null;

const CONFIG = window.CAMERA_CONFIG;
const INTERVAL = 1000 / CONFIG.sendFps;

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: CONFIG.canvasWidth },
                height: { ideal: CONFIG.canvasHeight },
                frameRate: { ideal: 30, max: 30 },
                facingMode: "user"
            },
            audio: false
        });

        video.srcObject = stream;
        running = true;
        statusBox.innerText = "Camera đang chạy, gửi frame về Flask...";

        if (timer) clearInterval(timer);
        timer = setInterval(sendFrame, INTERVAL);
    } catch (err) {
        statusBox.innerText = "Không mở được camera: " + err;
        console.error(err);
    }
}

function stopCamera() {
    running = false;

    if (timer) {
        clearInterval(timer);
        timer = null;
    }

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    labelBox.innerText = "STOPPED";
    labelBox.className = "result-label waiting";
    statusBox.innerText = "Camera đã dừng.";
}

async function resetAI() {
    await fetch("/reset_camera_ai", { method: "POST" });
    labelBox.innerText = "RESET";
    labelBox.className = "result-label waiting";
    scoresBox.innerText = "fusion=0.00 | lstm=0.00 | rule=0.00";
    infoBox.innerText = "persons=0 | latency=0ms";
    seqBox.innerText = `sequence=0/${CONFIG.seqLen}`;
    statusBox.innerText = "Đã reset bộ nhớ LSTM camera.";
}

async function sendFrame() {
    if (!running || sending) return;
    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    sending = true;

    try {
        canvas.width = CONFIG.canvasWidth;
        canvas.height = CONFIG.canvasHeight;

        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", CONFIG.jpegQuality));
        const formData = new FormData();
        formData.append("image", blob, "frame.jpg");

        const res = await fetch("/predict_frame", {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        if (data.error) {
            statusBox.innerText = "Server error: " + data.error;
            sending = false;
            return;
        }

        labelBox.innerText = data.label;
        if (data.label === "FIGHT") labelBox.className = "result-label fight";
        else if (data.label.startsWith("LOADING")) labelBox.className = "result-label loading";
        else labelBox.className = "result-label nofight";

        scoresBox.innerText =
            "fusion=" + data.fusion_score.toFixed(2) +
            " | lstm=" + data.lstm_score.toFixed(2) +
            " | rule=" + data.rule_score.toFixed(2);

        infoBox.innerText =
            "persons=" + data.person_count +
            " | interaction=" + data.interaction_score.toFixed(2) +
            " | motion=" + data.motion_score.toFixed(2) +
            " | FPS=" + data.fps.toFixed(1) +
            " | latency=" + data.latency_ms.toFixed(1) + "ms" +
            " | frame=" + data.frame_index;

        seqBox.innerText =
            "sequence=" + data.sequence_len + "/" + data.required_sequence +
            " | feature_count=" + data.feature_count;

        statusBox.innerText = "Đang xử lý realtime qua /predict_frame";
    } catch (err) {
        statusBox.innerText = "Lỗi gửi frame: " + err;
        console.error(err);
    }

    sending = false;
}
