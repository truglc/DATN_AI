import time
import numpy as np

class PerformanceMonitor:
    def __init__(self):
        self.times = []

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.times.append(time.time() - self.start_time)

    def fps(self):
        if len(self.times) == 0:
            return 0
        return 1 / np.mean(self.times)

    def latency(self):
        return np.mean(self.times) * 1000  # ms