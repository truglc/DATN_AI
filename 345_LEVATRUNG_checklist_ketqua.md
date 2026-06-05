# Checklist — Kết quả kiểm tra source

Tổng quan: danh sách kiểm tra các tính năng chính và trạng thái hiện tại của repository.

## ✅ Đã triển khai / Đã có
- [x] Sử dụng YOLO để phát hiện người trong video
- [x] Model Keras (LSTM) dự đoán bạo lực từ chuỗi frame
- [x] Pipeline upload video và trả về kết quả dự đoán
- [x] Demo web upload video và lưu trữ kết quả (Flask)
- [x] Mã huấn luyện CNN + LSTM trong `violence_detection.py`
- [x] DeepSORT tracking trong `FAPI/app/tracker.py`
- [x] Logic rule-based kết hợp tương tác người và dự đoán LSTM in `FAPI/app/behavior_fusion.py`
- [x] Temporal threshold / smoothing giảm false positive (`FAPI/app/temporal_filter.py`)
- [x] Phát hiện hành vi bất thường (té ngã, chạy, tĩnh) trong `FAPI/app/anomaly_detector.py`
- [x] Module đo hiệu năng: FPS / latency (`FAPI/app/performance.py`)

## 🚀 Đã hoàn thiện / Chạy trên Colab
- [x] Pipeline YOLO + DeepSORT + tracking: hoàn thiện, chạy trên Colab/GPU
- [x] Giao diện bounding box / tracking / cảnh báo realtime: hoàn thiện
- [x] Toàn bộ logic được triển khai và hoạt động trên môi trường Colab

## 📊 Đã kiểm chứng
- [x] Đánh giá precision / recall / F1 đầy đủ (`FAPI/app/metrics.py`)
- [x] Đo và ghi nhận FPS / latency thực tế
- [x] Kiểm thử chạy video trực tiếp từ camera và nhiều nguồn đầu vào



---
Last updated: cập nhật nội dung để dễ đọc hơn.

