# Violence Detection Project

Project Flask đã tách file:

```text
violence_detection_project/
├── app.py
├── requirements.txt
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── camera.html
│   ├── stream.html
│   ├── videos.html
│   ├── alerts.html
│   └── performance.html
├── static/
│   ├── css/style.css
│   └── js/camera.js
├── uploads/
├── outputs/
└── snapshots/
```

## Chạy local

```bash
pip install -r requirements.txt
python app.py
```

Mở:

```text
http://127.0.0.1:5000
```

## Model

Mặc định code đang dùng:

```python
MODEL_PATH = Path('/content/drive/MyDrive/model/best_violence_model.h5')
```

Nếu chạy local, sửa thành:

```python
MODEL_PATH = Path('outputs/best_violence_model.h5')
```

rồi đặt model vào thư mục `outputs/`.

## Các trang chính

```text
/                 Trang chủ
/camera_ai        Camera trình duyệt
/videos           Danh sách video upload
/alerts           Lịch sử cảnh báo
/performance      Hiệu năng
/webcam           Webcam server/local
```
