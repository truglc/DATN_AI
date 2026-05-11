import numpy as np
import cv2
from tensorflow.keras.models import load_model

class ViolenceModel:
    def __init__(self, window_size=16, step=4):
        self.model = load_model("models/prueba.h5")
        self.window_size = window_size
        self.step = step
        self.frames = []

    def predict(self, frame):
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0

        self.frames.append(frame)

        if len(self.frames) == self.window_size:
            data = np.expand_dims(np.array(self.frames), axis=0)
            pred = self.model.predict(data, verbose=0)[0][0]
            self.frames = self.frames[self.step:]
            return pred

        return None
