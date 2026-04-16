import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import models
import torch.nn as nn

# CNN trích đặc trưng
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Identity()
resnet.eval()

class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=16):
        self.samples = []
        self.num_frames = num_frames

        for label, folder in enumerate(["NonViolence", "Violence"]):
            folder_path = os.path.join(root_dir, folder)
            for file in os.listdir(folder_path):
                self.samples.append((os.path.join(folder_path, file), label))

    def __len__(self):
        return len(self.samples)

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // self.num_frames)

        for i in range(self.num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0
            frame = np.transpose(frame, (2, 0, 1))
            frames.append(frame)

        cap.release()

        frames = torch.tensor(frames).float()
        return frames

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        frames = self.extract_frames(video_path)

        features = []
        for frame in frames:
            frame = frame.unsqueeze(0)
            feat = resnet(frame)
            features.append(feat)

        features = torch.stack(features).squeeze(1)  # (16, 512)

        return features, label