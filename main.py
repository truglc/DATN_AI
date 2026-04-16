#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

# ======================
# MODEL (thay Classifier)
# ======================
class ViolenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# ======================
# MAIN
# ======================
class Main:
    def __init__(self):
        print("Initializing model...")
        self.model = ViolenceModel()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # giả lập dataset (bạn sẽ thay bằng dataset thật)
        self.train_loader = self._create_dummy_loader(200)
        self.val_loader = self._create_dummy_loader(50)
        self.test_loader = self._create_dummy_loader(50)

        self.bestThreshold = None

        # Time
        self._startTrainEpochTime = time.time()
        self._trainCountInOneEpoch = 0

        # config
        self.epochs = 5
        self.save_path = "checkpoints"

    def _create_dummy_loader(self, size):
        data = []
        for _ in range(size):
            x = torch.randn(16, 512)
            y = torch.randint(0, 2, (1,)).item()
            data.append((x, y))
        return data

    def Run(self):
        print("\nStart Training...\n")

        for epoch in range(self.epochs):
            self._trainCountInOneEpoch = 0
            start = time.time()

            # ===== TRAIN =====
            train_loss, train_acc = self.train()

            # ===== VALIDATION =====
            val_loss, val_acc = self.evaluate(self.val_loader)

            # ===== TEST =====
            test_loss, test_acc = self.evaluate(self.test_loader)

            print("Epoch:", epoch + 1)
            print(f"\t Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            print(f"\t Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
            print(f"\t Test  Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")
            print(f"\t Time: {time.time() - start:.2f}s\n")

            self.saveCheckpoint(epoch + 1)

        print("Optimization finished.")

    # ======================
    # TRAIN
    # ======================
    def train(self):
        self.model.train()
        total_loss = 0
        correct = 0

        for x, y in self.train_loader:
            x = x.unsqueeze(0).float()
            y = torch.tensor([y])

            output = self.model(x)
            loss = self.criterion(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            correct += (pred == y).sum().item()

        acc = correct / len(self.train_loader)
        return total_loss, acc

    # ======================
    # EVALUATE
    # ======================
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        correct = 0

        with torch.no_grad():
            for x, y in loader:
                x = x.unsqueeze(0).float()
                y = torch.tensor([y])

                output = self.model(x)
                loss = self.criterion(output, y)

                total_loss += loss.item()
                pred = torch.argmax(output, dim=1)
                correct += (pred == y).sum().item()

        acc = correct / len(loader)
        return total_loss, acc

    # ======================
    # SAVE MODEL
    # ======================
    def saveCheckpoint(self, epoch):
        os.makedirs(self.save_path, exist_ok=True)
        path = os.path.join(self.save_path, f"model_epoch_{epoch}.pt")
        torch.save(self.model.state_dict(), path)


# ======================
# RUN
# ======================
if __name__ == "__main__":
    main = Main()
    main.Run()