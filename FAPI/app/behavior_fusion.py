import numpy as np

class BehaviorFusion:
    def __init__(self, violence_threshold=0.7):
        self.th = violence_threshold
        self.track_memory = {}

    def update(self, track_id, lstm_prob, bbox_area):
        """
        bbox_area: diện tích người (lọc hành vi bất thường)
        """

        if track_id not in self.track_memory:
            self.track_memory[track_id] = []

        self.track_memory[track_id].append(lstm_prob)

        # giữ 10 frame gần nhất
        if len(self.track_memory[track_id]) > 10:
            self.track_memory[track_id].pop(0)

        avg_prob = np.mean(self.track_memory[track_id])

        # ================= RULE-BASED =================
        # rule 1: nếu bbox quá nhỏ → có thể là noise
        if bbox_area < 500:
            return 0.0

        # rule 2: cần ổn định nhiều frame mới báo động
        if len(self.track_memory[track_id]) < 5:
            return 0.0

        # rule 3: kết hợp LSTM + temporal smoothing
        if avg_prob > self.th:
            return avg_prob

        return avg_prob * 0.5  # giảm false positive