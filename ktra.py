import torch
import torch.nn as nn
import torch.nn.functional as F

# Tạo ảnh giả (ảnh số 5)
image = torch.tensor([
    [0,0,1,1,1,0],
    [0,0,1,0,0,0],
    [0,0,1,1,1,0],
    [0,0,0,0,1,0],
    [0,0,1,1,1,0],
    [0,0,0,0,0,0]
], dtype=torch.float32)

# reshape thành (batch, channel, H, W)
image = image.unsqueeze(0).unsqueeze(0)

print("INPUT:\n", image)

# ===== CNN với filter cố định =====
conv1 = nn.Conv2d(1, 2, kernel_size=3, bias=False)

# Gán filter cố định
# Filter 1: phát hiện đường ngang (pattern ngang)
conv1.weight.data[0,0] = torch.tensor([
    [1,1,1],
    [0,0,0],
    [0,0,0]
], dtype=torch.float32)

# Filter 2: phát hiện đường dọc (pattern dọc)
conv1.weight.data[1,0] = torch.tensor([
    [1,0,0],
    [1,0,0],
    [1,0,0]
], dtype=torch.float32)

# Conv
x = conv1(image)
print("\nSAU CONV1:\n", x)

# ReLU
x = F.relu(x)
print("\nSAU RELU:\n", x)

# Pooling
pool = nn.MaxPool2d(2,2)
x = pool(x)
print("\nSAU POOLING:\n", x)

# Flatten
x = x.view(x.size(0), -1)
print("\nSAU FLATTEN:\n", x)

# Dense layer (3 class)
fc = nn.Linear(x.shape[1], 3)
x = fc(x)
print("\nLOGITS:\n", x)

# Softmax
output = F.softmax(x, dim=1)
print("\nOUTPUT (XÁC SUẤT):\n", output)