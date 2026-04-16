# =========================
# 1. IMPORT LIBRARIES
# =========================
import os
import cv2
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed


# =========================
# 2. CONFIG
# =========================
IMG_SIZE = 64
SEQUENCE_LENGTH = 20


# =========================
# 3. LOAD DATASET
# =========================
def load_videos(dataset_path, max_videos_per_class=30):
    X, y = [], []
    classes = ['Violence', 'NonViolence']

    for label, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        videos = os.listdir(class_path)[:max_videos_per_class]

        print(f"Loading {class_name}...")

        for video in tqdm(videos):
            video_path = os.path.join(class_path, video)
            cap = cv2.VideoCapture(video_path)

            frames = []

            while len(frames) < SEQUENCE_LENGTH:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frame = frame / 255.0
                frames.append(frame)

            cap.release()

            if len(frames) == SEQUENCE_LENGTH:
                X.append(frames)
                y.append(label)

    return np.array(X), np.array(y)


# =========================
# 4. LOAD DATA
# =========================
dataset_path = "dataset"   # sửa path của bạn
X, y = load_videos(dataset_path)

print("DATA:", X.shape, y.shape)


# =========================
# 5. SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 6. BUILD MODEL (CNN + LSTM)
# =========================
model = Sequential()

model.add(TimeDistributed(
    Conv2D(32, (3,3), activation='relu'),
    input_shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)
))

model.add(TimeDistributed(MaxPooling2D(2,2)))
model.add(TimeDistributed(Flatten()))

model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# =========================
# 7. TRAIN MODEL
# =========================
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=4,
    validation_data=(X_test, y_test)
)


# =========================
# 8. SAVE MODEL
# =========================
model.save("violence_model.h5")
print("MODEL SAVED")


# =========================
# 9. LOAD MODEL (optional)
# =========================
# model = load_model("violence_model.h5")


# =========================
# 10. EVALUATION
# =========================
loss, acc = model.evaluate(X_test, y_test)
print("ACCURACY:", acc)


# =========================
# 11. PREDICT
# =========================
y_pred = (model.predict(X_test) > 0.5).astype("int32")


# =========================
# 12. CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()

print(classification_report(y_test, y_pred))


# =========================
# 13. TRAINING GRAPH
# =========================
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(["train", "val"])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.legend(["train", "val"])
plt.show()


# =========================
# 14. TEST VIDEO FUNCTION
# =========================
def predict_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)

    while len(frames) < SEQUENCE_LENGTH:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) == SEQUENCE_LENGTH:
        frames = np.array(frames)
        frames = np.expand_dims(frames, axis=0)

        pred = model.predict(frames)[0][0]

        if pred > 0.5:
            print("🔥 VIOLENCE:", pred)
        else:
            print("✅ NON-VIOLENCE:", pred)


# =========================
# 15. TEST SAMPLE VIDEO
# =========================
# predict_video("test.mp4")