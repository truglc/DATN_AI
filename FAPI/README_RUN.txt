Cách chạy:
1) Copy toàn bộ file trong gói này vào project.
2) Sửa MODEL_PATH trong app.py nếu cần.
3) Chạy: python app.py

Evaluate model:
python evaluate_model.py --model outputs/best_violence_model.h5 --test_npz outputs/test_features.npz --threshold 0.5 --positive_index 0

Các route mới:
- /settings
- /prediction_logs
- /outputs/<filename>
