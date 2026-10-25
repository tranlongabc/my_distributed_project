import io
import os
import sys
import time
import glob
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from flask import Flask, jsonify, request

# -----------------------------------------------------------------------------
# 1. CẤU HÌNH HỆ THỐNG & BIẾN MÔI TRƯỜNG (DÙNG CHO CẢ DOCKER VÀ THIẾT BỊ THẬT)
# -----------------------------------------------------------------------------
# Ép Python xả bộ nhớ đệm log ngay lập tức ra Terminal (ngăn trễ log)
sys.stdout.reconfigure(line_buffering=True)

# Tắt log HTTP rườm rà của Flask/Werkzeug
logging.getLogger('werkzeug').setLevel(logging.ERROR)
app = Flask(__name__)

# Tự động chọn thiết bị tính toán (CPU cho Raspberry Pi, CUDA/MPS nếu có GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cần hiểu thêm
# Địa chỉ mạng & Vai trò Node (Mặc định dùng tên service Docker, dễ dàng ghi đè bằng IP thật)
IP_NODE_2 = os.getenv("IP_NODE_2", "raspberry_pi_node_2")
IP_NODE_3 = os.getenv("IP_NODE_3", "raspberry_pi_node_3")
NODE_ROLE = os.getenv("NODE_ROLE", "node1").lower()
PORT = int(os.getenv("PORT", 5000))

CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "./data/checkpoints")
DATASET_DIR = os.getenv("DATASET_DIR", "./data")
LOG_INTERVAL = int(os.getenv("LOG_INTERVAL", 100))  # Mặc định in log mỗi 100 batch
MAX_BATCHES = int(os.getenv("MAX_BATCHES", 500))
# -----------------------------------------------------------------------------
# 2. KHỞI TẠO MÔ HÌNH MOBILENETV3-SMALL & CÁC LAYER PHÂN TÁN
# -----------------------------------------------------------------------------
print("Initializing MobileNetV3-Small model architecture")
full_model = models.mobilenet_v3_small(weights=None, num_classes=10)
features = full_model.features

# Phân rã mô hình thành 3 phần cho 3 Node
part1_net = nn.Sequential(*features[0:5]).to(DEVICE)
part2_net = nn.Sequential(*features[5:10]).to(DEVICE)
part3_net = nn.Sequential(
    *features[10:], full_model.avgpool, nn.Flatten(), full_model.classifier
).to(DEVICE)

# Optimizer & Loss Function
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.0001))
optimizer1 = optim.Adam(part1_net.parameters(), lr=LEARNING_RATE)
optimizer2 = optim.Adam(part2_net.parameters(), lr=LEARNING_RATE)
optimizer3 = optim.Adam(part3_net.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# -----------------------------------------------------------------------------
# 3. HTTP SESSION VỚI CƠ CHẾ RETRY (CHỐNG LỖI MẠNG TRÊN THIẾT BỊ THẬT)
# -----------------------------------------------------------------------------
def create_robust_session():
    """Tạo HTTP Session có khả năng tự thử lại khi mạng LAN/Wi-Fi bị chập chờn."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    return session

http_session = create_robust_session()

# -----------------------------------------------------------------------------
# 4. HÀM HỖ TRỢ SERIALIZE TENSOR & TẢI CHECKPOINT
# -----------------------------------------------------------------------------
def tensor_to_bytes(tensor):
    buffer = io.BytesIO()
    torch.save(tensor.cpu(), buffer)
    buffer.seek(0)
    return buffer.read()

def bytes_to_tensor(byte_data):
    buffer = io.BytesIO(byte_data)
    return torch.load(buffer, weights_only=False).to(DEVICE)

def package_to_bytes(data_dict):
    buffer = io.BytesIO()
    # Chuyển tensor trong dict về CPU trước khi serialize
    cpu_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}
    torch.save(cpu_dict, buffer)
    buffer.seek(0)
    return buffer.read()

def bytes_to_package(byte_data):
    buffer = io.BytesIO(byte_data)
    data = torch.load(buffer, weights_only=False)
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.to(DEVICE)
    return data

def load_checkpoints_safely():
    """Nạp checkpoint mới nhất tương ứng với vai trò của Node."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    role_map = {
        "node1": (part1_net, "part1_*.pth"),
        "node2": (part2_net, "part2_*.pth"),
        "node3": (part3_net, "part3_*.pth")
    }
    
    if NODE_ROLE in role_map:
        net, pattern = role_map[NODE_ROLE]
        ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, pattern))
        if ckpts:
            latest_ckpt = max(ckpts, key=os.path.getmtime)
            net.load_state_dict(torch.load(latest_ckpt, map_location=DEVICE))
            print(f"[{NODE_ROLE.upper()}] - Loaded checkpoint: {latest_ckpt}", flush=True)
        else:
            print(f"[{NODE_ROLE.upper()}] - No checkpoint found, starting from scratch.", flush=True)

# -----------------------------------------------------------------------------
# 5. XỬ LÝ CHO NODE 2 VÀ NODE 3 (FLASK PIPELINE SERVER)
# -----------------------------------------------------------------------------
if NODE_ROLE in ["node2", "node3"]:
    load_checkpoints_safely()
    request_counter = 0
    interval_loss_sum = 0.0
    counter_lock = threading.Lock()

    @app.route("/process", methods=["POST"])
    def process_layer():
        global request_counter, interval_loss_sum
        start_time = time.time()
        try:
            file_data = request.files["package"].read()
            package = bytes_to_package(file_data)
            
            input_tensor = package["tensor"].clone().detach().requires_grad_(True)
            labels = package["label"]
            
            # --- LUỒNG XỬ LÝ NODE 2 ---
            if NODE_ROLE == "node2":
                part2_net.train()
                output_tensor = part2_net(input_tensor)
                
                # Chuyển tiếp sang Node 3
                next_package = {"tensor": output_tensor, "label": labels}
                package_bytes = package_to_bytes(next_package)
                next_url = f"http://{IP_NODE_3}:{PORT}/process"
                
                response = http_session.post(next_url, files={"package": package_bytes}, timeout=30)
                resp_data = response.json()

                if resp_data.get("status") != "success":
                    return jsonify({"status": "error", "message": "Node 3 processing failed"}), 500
                
                grad_out2 = bytes_to_tensor(bytes.fromhex(resp_data["grad"]))
                
                # Backward & Optimizer
                optimizer2.zero_grad()
                output_tensor.backward(grad_out2)
                optimizer2.step()

                batch_loss = float(resp_data['loss'])
                torch.save(part2_net.state_dict(), os.path.join(CHECKPOINT_DIR, "part2_latest.pth"))
                elapsed = (time.time() - start_time) * 1000

                with counter_lock:
                    request_counter += 1
                    interval_loss_sum += batch_loss
                    current_count = request_counter
                    
                    if current_count % LOG_INTERVAL == 0:
                        avg_loss = interval_loss_sum / LOG_INTERVAL
                        print(f"[NODE2] Progress: {current_count} batches | Avg Loss ({LOG_INTERVAL} steps): {avg_loss:.4f} | Latency: {elapsed:.2f}ms", flush=True)
                        interval_loss_sum = 0.0
                    elif current_count == 1:
                        print(f"[NODE2] Progress: {current_count} batches | Loss: {batch_loss:.4f} | First Latency: {elapsed:.2f}ms", flush=True)

                return jsonify({
                    "status": "success",
                    "grad": tensor_to_bytes(input_tensor.grad).hex(),
                    "correct": resp_data["correct"],
                    "total": resp_data["total"],
                    "loss": resp_data["loss"]
                }), 200

            # --- LUỒNG XỬ LÝ NODE 3 ---
            elif NODE_ROLE == "node3":
                part3_net.train()
                outputs = part3_net(input_tensor)
                loss = criterion(outputs, labels)

                # Backward & Optimizer
                optimizer3.zero_grad()
                loss.backward()
                optimizer3.step()

                torch.save(part3_net.state_dict(), os.path.join(CHECKPOINT_DIR, "part3_latest.pth"))
                
                grad_out2 = input_tensor.grad
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels).sum().item()
                total = labels.size(0)
                
                batch_loss = float(loss.item())
                elapsed = (time.time() - start_time) * 1000

                with counter_lock:
                    request_counter += 1
                    interval_loss_sum += batch_loss
                    current_count = request_counter
                    
                    if current_count % LOG_INTERVAL == 0:
                        avg_loss = interval_loss_sum / LOG_INTERVAL
                        print(f"[NODE3] Progress: {current_count} batches | Avg Loss ({LOG_INTERVAL} steps): {avg_loss:.4f} | Correct: {correct}/{total} | Latency: {elapsed:.1f}ms", flush=True)
                        interval_loss_sum = 0.0
                    elif current_count == 1:
                        print(f"[NODE3] Progress: {current_count} batches | Loss: {batch_loss:.4f} | Correct: {correct}/{total} | First Latency: {elapsed:.1f}ms", flush=True)

                return jsonify({
                    "status": "success",
                    "grad": tensor_to_bytes(grad_out2).hex(),
                    "correct": correct,
                    "total": total,
                    "loss": batch_loss
                }), 200

        except Exception as e:
            print(f"Error Details: {e}", flush=True)
            return jsonify({"status": "error", "message": str(e)}), 500

# -----------------------------------------------------------------------------
# 6. XỬ LÝ CHO NODE 1 (ĐIỀU PHỐI DỮ LIỆU & HUẤN LUYỆN)
# -----------------------------------------------------------------------------
def run_node_1():
    print(f"\n[NODE1] Loading local CIFAR-10 dataset...", flush=True)
    load_checkpoints_safely()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Tự động tải dataset nếu chưa có sẵn
    download_flag = not os.path.exists(os.path.join(DATASET_DIR, "cifar-10-batches-py"))
    trainset = datasets.CIFAR10(root=DATASET_DIR, train=True, download=download_flag, transform=transform)
    trainloader = DataLoader(trainset, batch_size=int(os.getenv("BATCH_SIZE", 16)), shuffle=True)
    
    TOTAL_EPOCHS = int(os.getenv("EPOCHS", 1))
    print(f"[NODE1] Training info:", flush=True)
    print(f"         - Total batches / epoch: {len(trainloader)}", flush=True)
    print(f"         - Batch size: {trainloader.batch_size}", flush=True)
    print(f"         - Total epochs: {TOTAL_EPOCHS}", flush=True)
    print(f"[NODE1] Starting full-system fine-tuning", flush=True)
    time.sleep(10) 

    for epoch in range(TOTAL_EPOCHS):
        epoch_start_time = time.time()
        print(f"==================================================", flush=True)
        print(f"STARTING FINE-TUNE EPOCH {epoch+1}/{TOTAL_EPOCHS}", flush=True)
        print(f"==================================================", flush=True)
        
        total_correct = 0
        total_samples = 0
        running_loss = 0.0

        for i, (images, labels) in enumerate(trainloader):
            # --- ĐOẠN CODE KIỂM TRA GIỚI HẠN TEST NHANH ---
            if 'MAX_BATCHES' in globals() and MAX_BATCHES > 0 and i >= MAX_BATCHES:
                print(f"  [Node 1] Đã đạt giới hạn test nhanh ({MAX_BATCHES} batches). Dừng sớm epoch này.", flush=True)
                break
            # ---------------------------------------------
            batch_start = time.time()
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            try:
                part1_net.train()
                input_tensor = images.clone().detach().requires_grad_(True)
                
                # 1. Forward Pass tại Node 1
                out_part1 = part1_net(input_tensor)
                
                # 2. Gửi sang Node 2
                package = {"tensor": out_part1, "label": labels}
                package_bytes = package_to_bytes(package)
                target_url = f"http://{IP_NODE_2}:{PORT}/process"
                
                response = http_session.post(target_url, files={"package": package_bytes}, timeout=30)
                resp_data = response.json()

                if resp_data.get("status") == "success":
                    grad_out1 = bytes_to_tensor(bytes.fromhex(resp_data["grad"]))            
                    
                    # 3. Backward Pass & Optimizer tại Node 1
                    optimizer1.zero_grad()
                    out_part1.backward(grad_out1)
                    optimizer1.step()
                    
                    correct_batch = resp_data["correct"]
                    total_batch = resp_data["total"]
                    loss_batch = resp_data["loss"]
                    
                    total_correct += correct_batch
                    total_samples += total_batch
                    running_loss += loss_batch
                    
                    # In log mỗi LOG_INTERVAL batch
                    if (i + 1) % LOG_INTERVAL == 0 or (i + 1) == len(trainloader):
                        avg_loss = running_loss / (i + 1)
                        current_acc = (total_correct / total_samples) * 100
                        batch_time = (time.time() - batch_start) * 1000
                        print(f"  [NODE1] Epoch [{epoch+1}/{TOTAL_EPOCHS}] | Batch [{i+1}/{len(trainloader)}] | "
                              f"Avg Loss: {avg_loss:.4f} | Accuracy: {current_acc:.2f}% | Latency: {batch_time:.1f}ms", flush=True)
                    
                else:
                    print(f"  [NODE1 Warning] Response error at batch {i+1}: {resp_data}", flush=True)
                
            except Exception as e:
                print(f"  [NODE1 Connection Error] Batch {i+1}: {e}", flush=True)

        # Summary Epoch
        epoch_duration = time.time() - epoch_start_time
        epoch_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0
        epoch_loss = running_loss / len(trainloader) if len(trainloader) > 0 else 0.0
        
        print(f"\n--------------------------------------------------", flush=True)
        print(f"EPOCH {epoch+1}/{TOTAL_EPOCHS} COMPLETED", flush=True)
        print(f"Summary metrics:", flush=True)
        print(f"   - Average epoch loss: {epoch_loss:.4f}", flush=True)
        print(f"   - Accuracy: {epoch_acc:.2f}%", flush=True)
        print(f"   - Total samples processed: {total_samples}", flush=True)
        print(f"   - Epoch duration: {epoch_duration:.2f} seconds", flush=True)
        print(f"----------------------------------------------------\n", flush=True)
        
        ckpt_filename = os.path.join(CHECKPOINT_DIR, "part1_latest.pth")
        torch.save(part1_net.state_dict(), ckpt_filename)
        print(f"Safe checkpoint saved at: {ckpt_filename}\n", flush=True)

    print(f"THE FINE-TUNING PROCESS HAS BEEN FULLY COMPLETED!", flush=True)

# -----------------------------------------------------------------------------
# 7. MAIN ENTRYPOINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if NODE_ROLE == "node1":
        run_node_1()
    else:
        # Chạy Flask Server cho Node 2 và Node 3
        app.run(host="0.0.0.0", port=PORT, debug=False)