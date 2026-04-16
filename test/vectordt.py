import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ===== 1. Thư mục chứa frame =====
frames_dir = r"E:\rvideo\frames_16"
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

# ===== 2. Transform ảnh =====
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===== 3. Load CNN backbone =====
resnet = models.resnet18(pretrained=True)
resnet.eval()
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

# ===== 4. Chuyển frame thành feature vectors =====
feature_vectors = []

with torch.no_grad():
    for f in frame_files:
        img_path = os.path.join(frames_dir, f)
        img = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0)

        features = feature_extractor(img_tensor)
        features = features.view(features.size(0), -1)  # (1, 512)
        feature_vectors.append(features.squeeze(0))      # (512,)

# ===== 5. Chuyển sang tensor (n_frames, 512) =====
feature_vectors = torch.stack(feature_vectors)
print("Feature vectors shape:", feature_vectors.shape)

# ===== 6. Lưu vào txt =====
txt_path = r"E:\rvideo\frames_16_features.txt"
with open(txt_path, 'w') as f:
    for vec in feature_vectors:
        line = ' '.join([str(x.item()) for x in vec])
        f.write(line + '\n')

print(f"Lưu xong vector đặc trưng vào {txt_path}")