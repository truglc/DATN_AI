import torch
import cv2
import numpy as np
import torchvision.models as models
import torch.nn as nn

# ===== MODEL =====
class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(cnn.children())[:-1])
        self.lstm = nn.LSTM(512, 256, batch_first=True)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        b, t, c, h, w = x.size()

        x = x.view(b*t, c, h, w)
        x = self.cnn(x)
        x = x.view(b, t, -1)

        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])

        return x

# ===== LOAD VIDEO =====
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
frames = torch.tensor(frames).permute(0, 3, 1, 2)
frames = frames.unsqueeze(0)  # thêm batch

print("Input shape:", frames.shape)

# ===== MODEL =====
model = CNN_LSTM()
model.eval()

with torch.no_grad():
    out = model(frames)
    pred = torch.argmax(out, dim=1)

print("Output:", out)
print("Predict:", pred.item())


import torch.nn.functional as F

prob = F.softmax(out, dim=1)
print("Xác suất:", prob)