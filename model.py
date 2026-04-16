import torch
import torch.nn as nn

class ViolenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.lstm = nn.LSTM(input_size=512, hidden_size=128, batch_first=True)
        # self.fc = nn.Linear(128, 2)
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)