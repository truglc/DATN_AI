import torch
import torch.nn as nn

# giả lập 4 frame, mỗi frame 3 feature
x = torch.tensor([
    [0.1, 0.2, 0.3],   # frame 1
    [0.5, 0.4, 0.3],   # frame 2
    [0.9, 0.8, 0.7],   # frame 3
    [1.0, 0.9, 0.8]    # frame 4
], dtype=torch.float32)

# thêm batch
x = x.unsqueeze(0)  # (1, 4, 3)

lstm = nn.LSTM(input_size=3, hidden_size=5, batch_first=True)

output, (h_n, c_n) = lstm(x)

print("INPUT:\n", x)
print("\nOUTPUT từng bước:\n", output)
print("\nHIDDEN cuối:\n", h_n)
print("\nCELL STATE cuối:\n", c_n)