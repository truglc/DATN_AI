# evaluate_model.py

import argparse
import json
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())
    x_key = None
    y_key = None
    for k in ["X", "x", "features", "test_features", "arr_0"]:
        if k in keys:
            x_key = k
            break
    for k in ["y", "Y", "labels", "test_labels", "arr_1"]:
        if k in keys:
            y_key = k
            break
    if x_key is None or y_key is None:
        raise ValueError(f"Không tìm thấy X/y trong {path}. Keys hiện có: {keys}")
    return data[x_key], data[y_key]


def to_label(y):
    y = np.asarray(y)
    if y.ndim > 1 and y.shape[-1] > 1:
        return np.argmax(y, axis=-1)
    return y.reshape(-1).astype(int)


def pred_to_label(pred, fight_index=0, threshold=0.5):
    pred = np.asarray(pred)
    if pred.ndim == 1 or pred.shape[-1] == 1:
        return (pred.reshape(-1) >= threshold).astype(int)
    # Mặc định class index 0 là FIGHT theo app.py.
    # Để quy ước nhị phân: 1 = FIGHT, 0 = NO_FIGHT
    return (np.argmax(pred, axis=-1) == fight_index).astype(int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path tới .h5 model")
    parser.add_argument("--data", required=True, help="Path tới .npz test_features")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fight-index", type=int, default=0, help="Softmax index của lớp FIGHT")
    parser.add_argument("--out", default="evaluation_result.json")
    args = parser.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    print("Loading model:", model_path)
    model = load_model(str(model_path))
    print("Loading data:", data_path)
    X, y = load_npz(str(data_path))

    y_true_raw = to_label(y)
    pred = model.predict(X, verbose=1)
    y_pred = pred_to_label(pred, fight_index=args.fight_index, threshold=args.threshold)

    # Nếu y_true đang là softmax class 0/1 theo dataset, quy ước lại: 1 = FIGHT nếu class = fight_index.
    if y_true_raw.max() > 1 or len(np.unique(y_true_raw)) > 2:
        y_true = y_true_raw
    else:
        # Với train cũ nếu y_true=0 là FIGHT, đổi thành 1=FIGHT.
        y_true = (y_true_raw == args.fight_index).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n===== EVALUATION RESULT =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    result = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "model": str(model_path),
        "data": str(data_path),
        "threshold": args.threshold,
        "fight_index": args.fight_index
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\nSaved:", args.out)


if __name__ == "__main__":
    main()
