import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

# ======================
# LOAD MODELS
# ======================
model = load_model("violence_model.h5")

base = VGG16(weights='imagenet', include_top=True)
cnn_model = Model(inputs=base.input,
                  outputs=base.get_layer('fc2').output)

print("Models loaded!")

sequence_length = 20
frames = []

frame_count = 0
label = "Detecting..."
confidence = 0.0

# ======================
def extract_feature(frame):
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    feature = cnn_model.predict(img, verbose=0)
    return feature[0]

# ======================
def choose_video():
    file_path = filedialog.askopenfilename(
        filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
    )
    if file_path:
        process_video(file_path)

# ======================
def process_video(video_path):
    global frames, frame_count, label, confidence

    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ======================
        # SKIP FRAME (tăng tốc)
        # ======================
        if frame_count % 3 == 0:

            feature = extract_feature(frame)
            frames.append(feature)

            if len(frames) == sequence_length:

                input_data = np.expand_dims(frames, axis=0)

                pred = model.predict(input_data, verbose=0)[0][0]

                confidence = float(pred)

                if pred > 0.5:
                    label = "🔥 VIOLENCE DETECTED"
                else:
                    label = "NORMAL"

                frames.pop(0)

        # ======================
        # HIỂN THỊ TEXT LÊN VIDEO
        # ======================
        color = (0, 0, 255) if "VIOLENCE" in label else (0, 255, 0)

        cv2.putText(frame, label, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2)

        cv2.putText(frame, f"Confidence: {confidence:.2f}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)

        cv2.imshow("Violence Detection", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ======================
# UI
# ======================
root = tk.Tk()
root.title("Violence Detection")

btn = tk.Button(root, text="Chọn Video", command=choose_video,
                font=("Arial", 14), bg="blue", fg="white")

btn.pack(pady=40)

root.mainloop()