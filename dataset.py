import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import models
import torch.nn as nn

# ======================
# CNN FEATURE EXTRACTOR
# ======================
# resnet = models.resnet18(pretrained=True)
from torchvision.models import ResNet18_Weights
resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
resnet.fc = nn.Identity()   # bỏ layer cuối → lấy vector 512
resnet.eval()


# ======================
# DATASET
# ======================
class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=16):
        self.samples = []
        self.num_frames = num_frames

        # ⚠️ PHÙ HỢP DATASET CỦA BẠN
        folders = ["nofights", "fights"]

        for label, folder in enumerate(folders):
            folder_path = os.path.join(root_dir, folder)

            if not os.path.exists(folder_path):
                print(f"❌ Không tìm thấy thư mục: {folder_path}")
                continue

            for file in os.listdir(folder_path):
                if file.endswith((".avi", ".mp4", ".mov")):
                    full_path = os.path.join(folder_path, file)
                    self.samples.append((full_path, label))

        print(f"✅ Loaded {len(self.samples)} videos")

    def __len__(self):
        return len(self.samples)

    # ======================
    # EXTRACT FRAME
    # ======================
    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            cap.release()
            return None

        step = max(1, total // self.num_frames)

        for i in range(self.num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0
            frame = np.transpose(frame, (2, 0, 1))  # (H,W,C) → (C,H,W)

            frames.append(frame)

        cap.release()

        # nếu thiếu frame → pad thêm
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        # frames = torch.tensor(frames).float()  # (16, 3, 224, 224)
        frames = np.array(frames)
        frames = torch.from_numpy(frames).float()

        return frames

    # ======================
    # GET ITEM
    # ======================
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        frames = self.extract_frames(video_path)

        if frames is None:
            # fallback nếu video lỗi
            frames = torch.zeros(self.num_frames, 3, 224, 224)

        features = []

        # CNN → vector 512
        for frame in frames:
            frame = frame.unsqueeze(0)  # (1,3,224,224)

            with torch.no_grad():
                feat = resnet(frame)

            features.append(feat)

        features = torch.stack(features).squeeze(1)  # (16, 512)

        return features, label