# Checklist Kết quả Kiểm tra Source

## Đã triển khai / đã có
- [x] Sử dụng YOLO để phát hiện người trong video
- [x] Sử dụng model Keras (LSTM) để dự đoán bạo lực từ chuỗi frame
- [x] Có pipeline xử lý upload video và trả về kết quả dự đoán
- [x] Có module demo web upload video và lưu trữ kết quả (Flask/FastAPI)
- [x] Có mã huấn luyện mô hình CNN + LSTM trong `violence_detection.py`

## Chưa hoàn thành / cần bổ sung
- [ ] DeepSORT tracking thực sự chưa được triển khai (`FAPI/app/tracker.py` hiện chỉ là stub)
- [ ] Logic rule-based kết hợp tương tác người và dự đoán LSTM chưa có
- [ ] Cơ chế temporal threshold để giảm false positive chưa có
- [ ] Phát hiện hành vi bất thường ngoài bạo lực (té ngã, chạy bất thường) chưa có
- [ ] Đánh giá đầy đủ: precision, recall, F1-score chưa có
- [ ] Đánh giá hiệu năng: FPS và latency chưa đo lường
- [ ] Xử lý video trực tiếp từ camera / nhiều nguồn đầu vào chưa hoàn chỉnh
- [ ] Giao diện demo chưa hiển thị bounding box, tracking, cảnh báo realtime rõ ràng

## Ghi chú
- Nếu cần, có thể tiếp tục mở rộng `FAPI/app/main.py` và `FAPI/app/tracker.py` để thêm DeepSORT và logic rule-based.
- `violence_detection.py` đang tạo tiền xử lý VGG16 + LSTM, nhưng chưa kết nối với tracking và logic cảnh báo theo thời gian.
