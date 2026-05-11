class Tracker:
    def __init__(self, max_disappeared=30, max_distance=120):
        self.next_id = 1
        self.objects = {}  # track_id -> bbox
        self.disappeared = {}  # track_id -> disappeared frames count
        self.history = {}  # track_id -> list of history records
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _distance(self, c1, c2):
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        return (dx * dx + dy * dy) ** 0.5

    def register(self, bbox):
        track_id = self.next_id
        self.next_id += 1
        self.objects[track_id] = bbox
        self.disappeared[track_id] = 0
        centroid = self._centroid(bbox)
        self.history[track_id] = [{"bbox": bbox, "centroid": centroid}]

    def deregister(self, track_id):
        if track_id in self.objects:
            del self.objects[track_id]
        if track_id in self.disappeared:
            del self.disappeared[track_id]
        if track_id in self.history:
            del self.history[track_id]

    def update(self, detections):
        if len(detections) == 0:
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self.deregister(track_id)
            return []

        input_centroids = [self._centroid(box) for box in detections]

        if len(self.objects) == 0:
            for bbox in detections:
                self.register(bbox)
            return [
                {
                    "track_id": track_id,
                    "bbox": self.objects[track_id],
                    "history": self.history[track_id],
                }
                for track_id in self.objects
            ]

        object_ids = list(self.objects.keys())
        object_centroids = [self._centroid(self.objects[track_id]) for track_id in object_ids]

        distances = [
            [self._distance(object_centroid, input_centroid) for input_centroid in input_centroids]
            for object_centroid in object_centroids
        ]

        assigned_rows = set()
        assigned_cols = set()

        for row_idx in sorted(range(len(distances)), key=lambda r: min(distances[r])):
            if row_idx in assigned_rows:
                continue
            col_idx = min(range(len(distances[row_idx])), key=lambda c: distances[row_idx][c])
            if col_idx in assigned_cols:
                continue
            if distances[row_idx][col_idx] > self.max_distance:
                continue
            track_id = object_ids[row_idx]
            bbox = detections[col_idx]
            centroid = input_centroids[col_idx]
            self.objects[track_id] = bbox
            self.disappeared[track_id] = 0
            self.history[track_id].append({"bbox": bbox, "centroid": centroid})
            if len(self.history[track_id]) > 10:
                self.history[track_id].pop(0)
            assigned_rows.add(row_idx)
            assigned_cols.add(col_idx)

        for row_idx, track_id in enumerate(object_ids):
            if row_idx not in assigned_rows:
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self.deregister(track_id)

        for col_idx in range(len(detections)):
            if col_idx not in assigned_cols:
                self.register(detections[col_idx])

        return [
            {
                "track_id": track_id,
                "bbox": self.objects[track_id],
                "history": self.history[track_id],
            }
            for track_id in self.objects
        ]
