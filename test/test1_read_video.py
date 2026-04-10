import cv2

video_path = r"E:\rvideo\0H2s9UJcNJ0_4.avi"

cap = cv2.VideoCapture(video_path)

print("Mở video:", cap.isOpened())

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1

print("Tổng số frame:", count)

cap.release()