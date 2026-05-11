# FAPI/app/tracker.py

from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self):
        # Khởi tạo DeepSORT
        self.tracker = DeepSort(
            max_age=30,              # số frame giữ track khi mất detection
            n_init=3,                # số frame để xác nhận 1 track
            max_cosine_distance=0.4,
            nn_budget=100
        )

    def update(self, detections, frame):
        """
        detections format:
        [
            [x1, y1, x2, y2, confidence, class_id],
            ...
        ]

        frame: ảnh gốc (numpy array)
        """

        # DeepSORT yêu cầu format riêng
        ds_detections = []

        for det in detections:
            x1, y1, x2, y2, conf, cls = det

            width = x2 - x1
            height = y2 - y1

            ds_detections.append((
                [x1, y1, width, height],  # bbox
                conf,                     # confidence
                int(cls)                  # class id
            ))

        # Update tracker
        tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        results = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()  # left, top, right, bottom

            x1, y1, x2, y2 = map(int, ltrb)

            results.append({
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2]
            })

        return results