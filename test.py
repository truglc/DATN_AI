# import torch
# import os
# import cv2
# import numpy as np
# from model import ViolenceModel
# from torchvision import models
# import torch.nn as nn

# # ======================
# # DEVICE
# # ======================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ======================
# # LOAD MODEL
# # ======================
# model = ViolenceModel().to(device)
# model.load_state_dict(torch.load("best_model.pt", map_location=device))
# model.eval()

# # ======================
# # LOAD RESNET (CNN)
# # ======================
# resnet = models.resnet18(weights="IMAGENET1K_V1")
# resnet.fc = nn.Identity()
# resnet = resnet.to(device)
# resnet.eval()

# # ======================
# # EXTRACT FRAMES
# # ======================
# def extract_frames(video_path, num_frames=16):
#     cap = cv2.VideoCapture(video_path)
#     frames = []

#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     step = max(1, total // num_frames)

#     for i in range(num_frames):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame = cv2.resize(frame, (224, 224))
#         frame = frame / 255.0

#         mean = np.array([0.485, 0.456, 0.406])
#         std  = np.array([0.229, 0.224, 0.225])
#         frame = (frame - mean) / std

#         frame = np.transpose(frame, (2, 0, 1))
#         frames.append(frame)

#     cap.release()

#     if len(frames) == 0:
#         return torch.zeros(num_frames, 3, 224, 224)

#     return torch.tensor(frames).float()

# # ======================
# # PREDICT FUNCTION
# # ======================
# def predict(video_path):
#     frames = extract_frames(video_path).to(device)

#     with torch.no_grad():
#         features = resnet(frames)

#     features = features.unsqueeze(0)

#     with torch.no_grad():
#         output = model(features)
#         pred = torch.argmax(output, dim=1).item()

#     return pred

# # ======================
# # DATASET PATH
# # ======================
# root = r"D:\datn\archive\dataset\train\NonFight"

# classes = {
#     "Fight": 1,
#     "NonFight": 0
# }

# total = 0
# correct = 0

# # # ======================
# # # TEST LOOP
# # # ======================
# # for cls_name, label in classes.items():
# #     folder = os.path.join(root, cls_name)

# #     for file in os.listdir(folder):
# #         if not file.endswith((".avi", ".mp4", ".mov")):
# #             continue

# #         path = os.path.join(folder, file)

# #         try:
# #             pred = predict(path)

# #             total += 1
# #             if pred == label:
# #                 correct += 1

# #             print(f"{file} → pred: {pred}, true: {label}")

# #         except Exception as e:
# #             print(f"Skip error file: {file}")

# # # ======================
# # # RESULT
# # # ======================
# # acc = correct / total

# # print("\n======================")
# # print(f"TOTAL VIDEOS: {total}")
# # print(f"CORRECT: {correct}")
# # print(f"ACCURACY: {acc:.4f}")
# # print("======================")



# root = r"D:\datn\archive\dataset\train\NonFight"

# label = 0  # NonFight = 0

# total = 0
# correct = 0

# for file in os.listdir(root):
#     if not file.endswith((".avi", ".mp4", ".mov")):
#         continue

#     path = os.path.join(root, file)

#     try:
#         pred = predict(path)

#         total += 1
#         if pred == label:
#             correct += 1

#         print(f"{file} → pred: {pred}, true: {label}")

#     except:
#         print(f"Skip error file: {file}")

# acc = correct / total

# print("\n======================")
# print("NONFIGHT ONLY TEST")
# print(f"TOTAL: {total}")
# print(f"CORRECT: {correct}")
# print(f"ACCURACY: {acc:.4f}")
# print("======================")



import os

root = r"D:\DATN_AI\dataset"

fight = len(os.listdir(os.path.join(root, "Fight")))
nonfight = len(os.listdir(os.path.join(root, "NonFight")))

print("Fight:", fight)
print("NonFight:", nonfight)