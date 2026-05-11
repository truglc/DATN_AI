import numpy as np

class TemporalFilter:
    def __init__(self, window=15, threshold=0.7):
        self.window = window
        self.threshold = threshold
        self.buffer = {}

    def update(self, track_id, prob):
        if track_id not in self.buffer:
            self.buffer[track_id] = []

        self.buffer[track_id].append(prob)

        if len(self.buffer[track_id]) > self.window:
            self.buffer[track_id].pop(0)

        arr = np.array(self.buffer[track_id])

        # chỉ trigger nếu nhiều frame liên tiếp cao
        if np.mean(arr > self.threshold) > 0.6:
            return True, np.mean(arr)

        return False, np.mean(arr)