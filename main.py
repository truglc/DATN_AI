import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =========================
# CONFIG
# =========================
IMG_SIZE = 224
MAX_SEQ = 20   # số frame
MODEL_PATH = "D:\DATN_AI\FAPI\models\prueba.h5"

# =========================
# LOAD MODEL
# =========================
print("[INFO] Loading models...")
lstm_model = load_model(MODEL_PATH)

# CNN dùng extract feature
cnn_model = VGG16(weights="imagenet", include_top=True)
cnn_model = Model(inputs=cnn_model.input,
                  outputs=cnn_model.get_layer("fc2").output)

# =========================
# EXTRACT FRAME
# =========================
def get_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(total // MAX_SEQ, 1)

    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frames.append(frame)

        if len(frames) == MAX_SEQ:
            break

    cap.release()

    # padding nếu thiếu frame
    while len(frames) < MAX_SEQ:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))

    return np.array(frames)

# =========================
# EXTRACT FEATURE
# =========================
def extract_features(frames):
    features = []
    for f in frames:
        x = image.img_to_array(f)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        feat = cnn_model.predict(x, verbose=0)
        features.append(feat[0])

    return np.array(features)

# =========================
# PREDICT VIDEO
# =========================
def predict_video(video_path):
    print(f"[INFO] Processing: {video_path}")

    frames = get_frames(video_path)
    features = extract_features(frames)

    # reshape cho LSTM
    features = np.expand_dims(features, axis=0)  # (1, 20, 4096)

    pred = lstm_model.predict(features)[0][0]

    print(f"[RESULT] Score: {pred:.4f}")

    if pred > 0.5:
        return "🚨 BẠO LỰC"
    else:
        return "✅ KHÔNG BẠO LỰC"

# =========================
# TEST
# =========================
video_path = "test.mp4"

result = predict_video(video_path)
print("👉 Kết quả:", result)