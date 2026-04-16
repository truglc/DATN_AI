import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import ViolenceModel
from dataset import VideoDataset
import os

# ======================
# DEVICE
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# DATASET
# ======================
dataset = VideoDataset("dataset")

# split train / val (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# ======================
# MODEL
# ======================
model = ViolenceModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 10

best_val_acc = 0

os.makedirs("checkpoints", exist_ok=True)

# ======================
# TRAIN LOOP
# ======================
for epoch in range(epochs):

    # ===== TRAIN =====
    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0

    for x, y in train_loader:
        x = x.to(device).float()
        y = y.to(device).long()

        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        pred = torch.argmax(output, dim=1)
        train_correct += (pred == y).sum().item()
        train_total += y.size(0)

    train_acc = train_correct / train_total
    train_loss = train_loss / len(train_loader)

    # ===== VALIDATION =====
    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device).float()
            y = y.to(device).long()

            output = model(x)
            loss = criterion(output, y)

            val_loss += loss.item()

            pred = torch.argmax(output, dim=1)
            val_correct += (pred == y).sum().item()
            val_total += y.size(0)

    val_acc = val_correct / val_total
    val_loss = val_loss / len(val_loader)

    # ===== PRINT =====
    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    # ===== SAVE BEST MODEL =====
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "checkpoints/best_model.pt")
        print("✅ Best model saved!")

# ======================
print("\nTraining finished!")
print("Best Val Acc:", best_val_acc)