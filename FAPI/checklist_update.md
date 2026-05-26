# Checklist Update

## Đã triển khai / hiện có
- [x] Flask web app chính trong `app.py`
- [x] Upload video và lưu file trong thư mục `uploads/`
- [x] Xử lý video stream upload qua route `/video_feed/<video_id>`
- [x] Browser camera AI qua trang `/camera_ai` và endpoint `/predict_frame`
- [x] Tải YOLOv8 person detection từ `ultralytics`
- [x] Tải VGG16 feature extractor và sử dụng LSTM violence model inference
- [x] Logic fusion giữa `lstm_score` và `rule_score`
- [x] Temporal smoothing / threshold bằng `TemporalFilter`
- [x] Lưu cảnh báo snapshot vào `snapshots/`
- [x] SQLite database `database.db` với bảng `videos`, `alerts`, `performance`
- [x] Các route web hiển thị: `index`, danh sách `videos`, `alerts`, `performance`
- [x] Route `/webcam` cho webcam server/local
- [x] Route `/reset_camera_ai` để reset trạng thái camera
- [x] Performance logging mỗi `PERFORMANCE_LOG_EVERY` frame

## Cần hoàn thiện / thiếu
- [ ] Chưa có file model `.h5` trong repo để chạy local; thư mục `outputs/` trống
- [ ] `MODEL_PATH` hiện mặc định là đường dẫn Colab (`/content/drive/MyDrive/...`) cần đổi nếu chạy local
- [ ] `USE_DEEPSORT = False` nên DeepSORT chưa thực sự được dùng trong pipeline
- [ ] Không có các module `violence_detection.py`, `FAPI/app/tracker.py`, `FAPI/app/behavior_fusion.py`, `FAPI/app/temporal_filter.py`, `FAPI/app/anomaly_detector.py`, `FAPI/app/metrics.py` trong repo hiện tại
- [ ] Chưa có script huấn luyện model rõ ràng trong repo hiện tại
- [ ] Camera browser đang chạy fast mode chỉ dùng LSTM, chưa có tracking/YOLO thực sự cho camera

## Gợi ý cải thiện
- [ ] Sửa `MODEL_PATH` về `outputs/best_violence_model.h5` và thêm file model tương ứng
- [ ] Nếu cần DeepSORT tracking thật sự, bật `USE_DEEPSORT = True` và kiểm tra dependency `deep_sort_realtime`
- [ ] Tách logic inference/processing ra module riêng cho dễ bảo trì
- [ ] Thêm route hoặc trang hướng dẫn để người dùng biết cách upload, xem alerts và performance
- [ ] Kiểm tra lại các template HTML để đảm bảo tương thích với route hiện có

## Kết luận
Repo hiện tại có một pipeline Flask khá hoàn chỉnh cho upload video, camera browser, LSTM inference và alert logging. Tuy nhiên checklist cũ mô tả nhiều module không tồn tại trong repo này và cần cập nhật lại khi model local hoặc mô-đun huấn luyện được thêm vào.`