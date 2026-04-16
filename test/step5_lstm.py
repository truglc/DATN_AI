import torch
import torch.nn as nn

# lấy output từ CNN
from step4_cnn_extract import features   # (16, 512)

# ===== Chuẩn hóa input cho LSTM =====
# thêm batch dimension
features = features.unsqueeze(0)   # (1, 16, 512)

print("Input LSTM:", features.shape)


# ===== LSTM MODEL =====
class LSTM_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # x: (batch, seq, 512)

        out, _ = self.lstm(x)

        # out: (batch, seq, 128)
        print("LSTM output:", out.shape)

        # lấy timestep cuối
        out = out[:, -1, :]

        # (batch, 128)
        out = self.fc(out)

        return out


# ===== TEST =====
model = LSTM_Model()

output = model(features)

print("Final output:", output.shape)