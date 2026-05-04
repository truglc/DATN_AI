from ultralytics import YOLO

class Detector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def detect(self, frame):
        results = self.model(frame)[0]
        boxes = []

        for box in results.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))

        return boxes