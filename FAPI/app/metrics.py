class MetricsTracker:
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def update(self, y_true, y_pred):
        if y_true == 1 and y_pred == 1:
            self.tp += 1
        elif y_true == 0 and y_pred == 0:
            self.tn += 1
        elif y_true == 0 and y_pred == 1:
            self.fp += 1
        elif y_true == 1 and y_pred == 0:
            self.fn += 1

    def precision(self):
        return self.tp / (self.tp + self.fp + 1e-6)

    def recall(self):
        return self.tp / (self.tp + self.fn + 1e-6)

    def f1(self):
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r + 1e-6)

    def report(self):
        return {
            "precision": float(self.precision()),
            "recall": float(self.recall()),
            "f1": float(self.f1()),
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn
        }