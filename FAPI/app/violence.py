import numpy as np
import cv2
from tensorflow.keras.models import load_model

class ViolenceModel:
    def __init__(self):
        self.model = load_model("models/prueba.h5")
        self.frames = []

    def predict(self, frame):
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0

        self.frames.append(frame)

        if len(self.frames) == 16:
            data = np.expand_dims(self.frames, axis=0)
            pred = self.model.predict(data)[0][0]
            self.frames = []
            return pred

        return None