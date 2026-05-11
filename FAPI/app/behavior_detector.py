from collections import deque
import math


class BehaviorDetector:
    """
    Phát hiện hành vi bất thường dựa trên lịch sử bounding box:
    - Fall Detected (té ngã)
    - Abnormal Running (chạy bất thường)
    """

    def __init__(
        self,
        max_history=30,
        fall_ratio_threshold=1.2,
        fall_frames_required=10,
        running_speed_threshold=20.0
    ):
        # Lưu lịch sử bbox của từng track_id
        self.history = {}

        # Số frame lưu lại cho mỗi người
        self.max_history = max_history

        # Ngưỡng width/height để xác định đang nằm
        self.fall_ratio_threshold = fall_ratio_threshold

        # Cần bao nhiêu frame liên tiếp có ratio lớn
        self.fall_frames_required = fall_frames_required

        # Ngưỡng tốc độ (pixel/frame)
        self.running_speed_threshold = running_speed_threshold

    def update(self, track_id, bbox):
        """
        Cập nhật lịch sử và trả về nhãn hành vi.

        Parameters
        ----------
        track_id : int
            ID của đối tượng do tracker gán.
        bbox : list hoặc tuple
            [x1, y1, x2, y2]

        Returns
        -------
        str
            "Normal"
            "Fall Detected"
            "Abnormal Running"
        """
        x1, y1, x2, y2 = bbox

        w = x2 - x1
        h = y2 - y1

        # Tránh chia cho 0
        if h <= 0:
            return "Normal"

        # Tâm bounding box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Khởi tạo lịch sử nếu chưa có
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.max_history)

        # Lưu thông tin frame hiện tại
        self.history[track_id].append({
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h
        })

        # Phân tích hành vi
        return self.detect(track_id)

    def detect(self, track_id):
        data = self.history[track_id]

        # Chưa đủ dữ liệu
        if len(data) < 5:
            return "Normal"

        # =====================================================
        # 1. FALL DETECTION
        # =====================================================
        # Nếu width / height > 1.2 => người có thể đang nằm
        recent_items = list(data)[-15:]
        fall_count = 0

        for item in recent_items:
            ratio = item["w"] / max(item["h"], 1)
            if ratio > self.fall_ratio_threshold:
                fall_count += 1

        if fall_count >= self.fall_frames_required:
            return "Fall Detected"

        # =====================================================
        # 2. RUNNING DETECTION
        # =====================================================
        speeds = []

        for i in range(1, len(data)):
            dx = data[i]["cx"] - data[i - 1]["cx"]
            dy = data[i]["cy"] - data[i - 1]["cy"]

            speed = math.sqrt(dx * dx + dy * dy)
            speeds.append(speed)

        if len(speeds) > 0:
            recent_speeds = speeds[-10:]
            avg_speed = sum(recent_speeds) / len(recent_speeds)

            if avg_speed > self.running_speed_threshold:
                return "Abnormal Running"

        # =====================================================
        # 3. NORMAL
        # =====================================================
        return "Normal"

    def remove_track(self, track_id):
        """
        Xóa lịch sử của track khi đối tượng biến mất.
        """
        if track_id in self.history:
            del self.history[track_id]

    def reset(self):
        """
        Xóa toàn bộ lịch sử.
        """
        self.history.clear()