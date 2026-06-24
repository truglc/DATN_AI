# Checklist Kết quả Kiểm tra Source

## Đã triển khai / đã có
- [x] Sử dụng YOLO để phát hiện người trong video
- [x] Sử dụng model Keras (LSTM) để dự đoán bạo lực từ chuỗi frame
- [x] Có pipeline xử lý upload video và trả về kết quả dự đoán
- [x] Có module demo web upload video và lưu trữ kết quả (Flask)
- [x] Có mã huấn luyện mô hình CNN + LSTM trong `violence_detection.py`
- [x] Có module DeepSORT tracking thật sự trong `FAPI/app/tracker.py`
- [x] Có logic rule-based kết hợp tương tác người và dự đoán LSTM trong `FAPI/app/behavior_fusion.py`
- [x] Có cơ chế temporal threshold / smoothing để giảm false positive trong `FAPI/app/temporal_filter.py`
- [x] Có phát hiện hành vi bất thường ngoài bạo lực (té ngã, chạy bất thường, tĩnh) trong `FAPI/app/anomaly_detector.py`
- [x] Có module đo hiệu năng FPS / latency trong `FAPI/app/performance.py`

## Đã hoàn thiện / đang chạy trên Colab
- [x] Pipeline YOLO + DeepSORT + tracking đã hoàn thiện và đang chạy trên Colab/GPU
- [x] Giao diện bounding box / tracking / cảnh báo realtime đã hoàn thiện, không còn tạm giữ do CPU quá tải
- [x] Toàn bộ logic đã được triển khai và hoạt động trên môi trường Colab

## Đã hoàn thiện / đã kiểm chứng
- [x] Tích hợp đánh giá precision / recall / F1 đầy đủ (`FAPI/app/metrics.py` đã hoàn chỉnh và nối vào pipeline)
- [x] Đo số liệu hiệu năng thực tế: FPS và latency đã ghi nhận bằng kết quả chạy thật
- [x] Kiểm tra chạy video trực tiếp từ camera / nhiều nguồn đầu vào bằng pipeline hoàn chỉnh

## Ghi chú
- Các module và logic chính đã có, và pipeline đã chạy tốt trên Colab/GPU.
- Hiện tại không còn tạm giữ do CPU; Colab GPU đã giúp chạy full pipeline YOLO + DeepSORT + temporal filter.
