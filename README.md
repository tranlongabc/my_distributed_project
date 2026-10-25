# 🚀 Distributed AI Training on Resource-Constrained Embedded Clusters

Hệ thống huấn luyện và kiểm thử phân tán 3 Node sử dụng mô hình **MobileNetV3-Small** trên tập dữ liệu **CIFAR-10**. Dự án được tối ưu hóa theo mô hình Pipeline Parallelism, cho phép chạy mượt mà trên môi trường **Docker Container** (tách biệt terminal) cũng như chuyển đổi trực tiếp sang **cụm phần cứng nhúng thực tế (Raspberry Pi)** qua mạng LAN.

🛠️ Hướng dẫn vận hành
- Mở 3 cửa sổ Terminal tại thư mục gốc và chạy theo thứ tự
docker-compose up raspberry_pi_node_3
docker-compose up raspberry_pi_node_2
docker-compose up raspberry_pi_node_1

## 📂 Cấu trúc dự án (Project Layout)
```text
my_distributed_project/
├── client_code/
│   └── model_pipeline.py    # Mã nguồn chính xử lý huấn luyện & kiểm thử phân tán
├── data/
│   ├── checkpoints/         # Lưu trữ các file trọng số (.pth) cho 3 Node
│   │   ├── part1_latest.pth
│   │   ├── part2_latest.pth
│   │   └── part3_latest.pth
│   └── cifar-10-batches-py/ # Tập dữ liệu CIFAR-10 (tự động tải nếu chưa có)
├── docker-compose.yml       # Cấu hình orchestration cho cụm 3 Docker Container
├── Dockerfile               # File đóng gói môi trường Python & PyTorch
├── requirements.txt         # Thư viện phụ thuộc
└── README.md                # Hướng dẫn sử dụng dự án



