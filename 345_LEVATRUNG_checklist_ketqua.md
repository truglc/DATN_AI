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

## Đã định nghĩa và đang tối ưu / tạm giữ do CPU quá tải
- [x] Pipeline YOLO + DeepSORT + tracking đã định nghĩa, hiện đang tạm giữ để giảm tải CPU
- [x] Giao diện bounding box / tracking / cảnh báo realtime đã định nghĩa, một số phần đang tạm giữ để tối ưu hiệu năng
- [x] Một số phần logic đã định nghĩa sẵn để dùng khi hệ thống đủ tài nguyên

## Cần hoàn thiện / kiểm chứng thêm
- [ ] Tích hợp đánh giá precision / recall / F1 đầy đủ (`FAPI/app/metrics.py` cần hoàn chỉnh và nối vào pipeline)
- [ ] Đo số liệu hiệu năng thực tế: FPS và latency chưa được ghi nhận bằng kết quả chạy thật
- [ ] Kiểm tra chạy video trực tiếp từ camera / nhiều nguồn đầu vào bằng pipeline hoàn chỉnh

## Ghi chú
- Các module và logic chính đã có, nhưng vì tối ưu hiệu năng trên CPU nên hiện tại chưa bật full pipeline.
- Khi hệ thống đủ tài nguyên, chỉ cần bật lại YOLO + DeepSORT + rule-based + temporal filter là có thể chạy end-to-end.
