import cv2
import torch
import numpy as np

cap = cv2.VideoCapture(r"E:\rvideo\0H2s9UJcNJ0_4.avi")

frames = []

for i in range(16):
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    frames.append(frame)

cap.release()

frames = np.array(frames, dtype=np.float32)

print("Shape ban đầu:", frames.shape)

frames = torch.tensor(frames).permute(0, 3, 1, 2)

print("Sau khi convert:", frames.shape)