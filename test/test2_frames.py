import cv2

cap = cv2.VideoCapture(r"E:\rvideo\0H2s9UJcNJ0_4.avi")

for i in range(16):
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Frame", frame)
    cv2.waitKey(200)  # 200ms

cap.release()
cv2.destroyAllWindows()