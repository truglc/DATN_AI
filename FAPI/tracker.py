import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load model
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ======================
        # YOLO DETECTION
        # ======================
        results = model(frame)[0]

        detections = []
        for r in results.boxes.data:
            x1, y1, x2, y2, conf, cls = r.tolist()
            detections.append(([x1, y1, x2-x1, y2-y1], conf, int(cls)))

        # ======================
        # DEEPSORT TRACKING
        # ======================
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, w_box, h_box = track.to_ltrb()

            x1, y1, x2, y2 = int(l), int(t), int(w_box), int(h_box)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        out.write(frame)

    cap.release()
    out.release()