import numpy as np

class AnomalyDetector:
    def __init__(self):
        self.speed_memory = {}

    def update(self, track_id, centroid):
        """
        centroid: (x, y)
        """

        if track_id not in self.speed_memory:
            self.speed_memory[track_id] = []

        self.speed_memory[track_id].append(centroid)

        if len(self.speed_memory[track_id]) < 5:
            return "normal"

        pts = self.speed_memory[track_id]

        # tính tốc độ
        dx = pts[-1][0] - pts[-2][0]
        dy = pts[-1][1] - pts[-2][1]
        speed = (dx**2 + dy**2) ** 0.5

        # ================= RULES =================

        # 1. chạy bất thường
        if speed > 80:
            return "running_anomaly"

        # 2. đứng yên lâu
        if np.std(pts[-5:], axis=0).mean() < 2:
            return "static_anomaly"

        # 3. té ngã (bbox thay đổi nhanh theo chiều dọc)
        if abs(dy) > 50:
            return "fall"

        return "normal"