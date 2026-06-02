import argparse
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model


def find_key(data, candidates):
    for key in candidates:
        if key in data:
            return key
    raise KeyError(f"Không tìm thấy key trong npz. Cần một trong: {candidates}. Key hiện có: {list(data.keys())}")


def to_binary_labels(y, positive_index=0):
    y = np.asarray(y)

    if y.ndim == 2 and y.shape[1] > 1:
        return (np.argmax(y, axis=1) == positive_index).astype(int)

    return y.reshape(-1).astype(int)


def prediction_scores(pred, positive_index=0):
    pred = np.asarray(pred)

    if pred.ndim == 1:
        return pred

    if pred.ndim == 2 and pred.shape[1] == 1:
        return pred[:, 0]

    return pred[:, positive_index]


def confusion_matrix_binary(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tn, fp, fn, tp


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN/VGG16 + LSTM violence model")
    parser.add_argument("--model", required=True, help="Đường dẫn file .h5 model")
    parser.add_argument("--test_npz", required=True, help="Đường dẫn test_features.npz")
    parser.add_argument("--threshold", type=float, default=0.5, help="Ngưỡng phân loại binary/sigmoid")
    parser.add_argument("--positive_index", type=int, default=0, help="Với softmax 2 lớp: index của lớp FIGHT")
    parser.add_argument("--output_dir", default="eval_outputs", help="Thư mục lưu metrics.txt")
    args = parser.parse_args()

    model_path = Path(args.model)
    test_npz = Path(args.test_npz)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Không thấy model: {model_path}")

    if not test_npz.exists():
        raise FileNotFoundError(f"Không thấy file test npz: {test_npz}")

    data = np.load(test_npz)

    x_key = find_key(data, ["X", "x", "features", "test_features", "arr_0"])
    y_key = find_key(data, ["y", "Y", "labels", "test_labels", "arr_1"])

    X = np.asarray(data[x_key]).astype(np.float32)
    y_raw = np.asarray(data[y_key])

    y_true = to_binary_labels(y_raw, positive_index=args.positive_index)

    print("Loading model:", model_path)
    model = load_model(str(model_path))

    print("X shape:", X.shape)
    print("y shape:", y_raw.shape)

    pred = model.predict(X, verbose=1)
    scores = prediction_scores(pred, positive_index=args.positive_index)

    y_pred = (scores >= args.threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix_binary(y_true, y_pred)

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    result = f"""
===== MODEL EVALUATION =====
Model: {model_path}
Test file: {test_npz}
Feature key: {x_key}
Label key: {y_key}
Threshold: {args.threshold}
Positive index: {args.positive_index}

Accuracy : {accuracy:.4f}
Precision: {precision:.4f}
Recall   : {recall:.4f}
F1-score : {f1:.4f}

Confusion matrix:
TN={tn}  FP={fp}
FN={fn}  TP={tp}
"""

    print(result)

    metrics_path = output_dir / "metrics.txt"
    metrics_path.write_text(result, encoding="utf-8")
    print("Saved:", metrics_path)


if __name__ == "__main__":
    main()
