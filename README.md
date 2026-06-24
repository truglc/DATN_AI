# Violence Detection Complete Project - Toggle YOLO/DeepSORT

## Chức năng chính

- Upload video và stream kết quả trên web.
- Browser Camera gửi frame về Flask qua `/predict_frame`.
- CNN/VGG16 `fc2=4096` + LSTM sequence 20 để nhận diện `FIGHT / NO FIGHT`.
- YOLO detect người, DeepSORT tracking ID người.
- Rule-based hỗ trợ: distance, IoU, motion, fall score, running score.
- Lưu cảnh báo, prediction logs, FPS, latency vào SQLite.
- Lưu video output đã vẽ overlay.
- Trang cấu hình realtime `/config`.

## Logic nhãn

Output cuối cùng chỉ có:

- `FIGHT`
- `NO FIGHT`

`fall_score` và `running_score` không phải nhãn riêng. Chúng chỉ là điểm phụ để tăng `rule_score` trong ngữ cảnh đánh nhau/bạo lực.

## Bật/tắt YOLO và DeepSORT

Vào trang:

```txt
http://127.0.0.1:5000/config
```

Có các công tắc:

- `USE_YOLO`: bật/tắt YOLO cho upload video.
- `USE_DEEPSORT`: bật/tắt DeepSORT tracking. Nếu tắt DeepSORT nhưng YOLO còn bật, hệ thống vẫn dùng bbox YOLO và gán ID tạm.
- `CAMERA_USE_YOLO`: bật/tắt YOLO cho browser camera. Nên tắt để camera nhanh hơn.
- `SAVE_OUTPUT_VIDEO`: bật/tắt lưu video output.

Khi `USE_YOLO = False`:

- Không detect người.
- Không vẽ bbox.
- Không tính IoU/distance/fall/running.
- Hệ thống vẫn chạy CNN+LSTM + `motion_score` rule cơ bản.

Công thức khi bật YOLO:

```txt
rule_score = 0.35*interaction + 0.25*iou + 0.20*motion + 0.20*danger_score
fusion_score = LSTM_WEIGHT*lstm_score + RULE_WEIGHT*rule_score
```

Công thức khi tắt YOLO:

```txt
rule_score = motion_score
fusion_score = LSTM_WEIGHT*lstm_score + RULE_WEIGHT*rule_score
```

## Cài thư viện

```bash
pip install flask opencv-python numpy tensorflow ultralytics deep-sort-realtime scikit-learn matplotlib
```

Nếu không dùng DeepSORT:

```bash
pip install flask opencv-python numpy tensorflow ultralytics scikit-learn matplotlib
```

## Chạy

```bash
python app.py
```

Mở:

```txt
http://127.0.0.1:5000
```

## Model path

Mặc định:

```txt
/content/drive/MyDrive/model/best_violence_model.h5
```

Có thể sửa trong `app.py` hoặc set biến môi trường:

```bash
export MODEL_PATH=/duong/dan/model.h5
```

Trên Windows PowerShell:

```powershell
$env:MODEL_PATH="E:\data\best_violence_model.h5"
python app.py
```

## Đánh giá model

```bash
python evaluate_model.py --features /content/outputs/test_features.npz --model /content/outputs/best_violence_model.h5
```

Kết quả gồm Accuracy, Precision, Recall, F1 và confusion matrix.
