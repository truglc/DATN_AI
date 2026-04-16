import cv2
import os

video_path = r"E:\rvideo\fi1_xvid.avi"
output_folder = "frames_all"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    filename = os.path.join(output_folder, f"frame_{count:04d}.jpg")
    cv2.imwrite(filename, frame)

    count += 1

cap.release()

print("Tổng frame:", count)