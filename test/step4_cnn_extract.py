import torch
import torch.nn as nn
import torchvision.models as models

# Load tensor từ bước trước
from step3_load_to_tensor import tensor   # (16, 3, 224, 224)

# ===== CNN MODEL =====
class CNN_Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        model = models.resnet18(pretrained=True)

        # bỏ layer cuối (fc)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.feature_extractor(x)

        # (batch, 512, 1, 1) → (batch, 512)
        x = x.view(x.size(0), -1)

        return x

# ===== TEST =====
cnn = CNN_Encoder()

features = cnn(tensor)

print("Feature shape:", features.shape)