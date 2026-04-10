import os
import cv2
import torch
import numpy as np

folder = "frames_16"

frames = []

files = sorted(os.listdir(folder))

for file in files:
    path = os.path.join(folder, file)

    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0

    # (H, W, C) → (C, H, W)
    img = np.transpose(img, (2, 0, 1))

    frames.append(img)

frames = np.array(frames)

tensor = torch.tensor(frames, dtype=torch.float32)

print("Shape:", tensor.shape)