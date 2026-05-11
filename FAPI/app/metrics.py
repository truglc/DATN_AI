import numpy as np

class Metrics:
    def __init__(self):
        self.y_true = []
        self.y_pred = []

    def update(self, true, pred):
        self.y_true.append(true)
        self.y_pred.append(pred)

    def precision(self):
        tp = sum((np.array(self.y_true)==1) & (np.array(self.y_pred)==1))
        fp = sum((np.array(self.y_true)==0) & (np.array(self.y_pred)==1))
        return tp / (tp + fp + 1e-6)

    def recall(self):
        tp = sum((np.array(self.y_true)==1) & (np.array(self.y_pred)==1))
        fn = sum((np.array(self.y_true)==1) & (np.array(self.y_pred)==0))
        return tp / (tp + fn + 1e-6)

    def f1(self):
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r + 1e-6)