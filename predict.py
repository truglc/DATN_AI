import torch
import cv2
import numpy as np
from model import ViolenceModel
from torchvision import models
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model = ViolenceModel().to(device)
model.load_state_dict(torch.load("checkpoints/model.pt", map_location=device))
model.eval()

# CNN
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Identity()
resnet.eval()

def extract(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // num_frames)

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        frames.append(frame)

    cap.release()
    return torch.tensor(frames).float()

def predict(video_path):
    frames = extract(video_path)

    features = []
    for frame in frames:
        frame = frame.unsqueeze(0)
        feat = resnet(frame)
        features.append(feat)

    features = torch.stack(features).squeeze(1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(features)
        pred = torch.argmax(output, dim=1).item()

    return "Violence" if pred == 1 else "Normal"

# test
video = "test.mp4"
print(predict(video))