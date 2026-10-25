import io
import os
import time
import glob
import requests
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from flask import Flask, jsonify, request
import logging

# Tắt log truy cập HTTP mặc định của Werkzeug (Flask development server)
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.ERROR)
app = Flask(__name__)

# --- CẤU HÌNH HỆ THỐNG & ĐỊA CHỈ MẠNG ---
IP_NODE_2 = os.getenv("IP_NODE_2", "raspberry_pi_node_2")
IP_NODE_3 = os.getenv("IP_NODE_3", "raspberry_pi_node_3")
NODE_ROLE = os.getenv("NODE_ROLE", "node1")

CHECKPOINT_DIR = "./data/checkpoints"




# Bắt đầu hiểu
# KHỞI TẠO MÔ HÌNH MOBILENETV3-SMALL (CIFAR-10: 10 Classes)
print("Initializing MobileNetV3-Small model architecture")
full_model = models.mobilenet_v3_small(weights=None, num_classes=10)
features = full_model.features

# PHÂN RÃ MÔ HÌNH THÀNH 3 PHẦN TUẦN TỰ CHO 3 NODE
part1_net = nn.Sequential(*features[0:5])
part2_net = nn.Sequential(*features[5:10])
part3_net = nn.Sequential(
    *features[10:], full_model.avgpool, nn.Flatten(), full_model.classifier
)
# Kết thúc hiểu





# --- CẤU HÌNH OPTIMIZER & LOSS FUNCTION (FINE-TUNING LR = 0.0001) ---
LEARNING_RATE = 0.0001
optimizer1 = optim.Adam(part1_net.parameters(), lr=LEARNING_RATE)
optimizer2 = optim.Adam(part2_net.parameters(), lr=LEARNING_RATE)
optimizer3 = optim.Adam(part3_net.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()



# Bắt đầu hiểu
# CÁC HÀM HỖ TRỢ CHUYỂN ĐỔI TENSOR / BYTE 
def tensor_to_bytes(tensor):
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    return buffer.read()

def bytes_to_tensor(byte_data):
    buffer = io.BytesIO(byte_data)
    return torch.load(buffer, weights_only=False)

def package_to_bytes(data_dict):
    buffer = io.BytesIO()
    torch.save(data_dict, buffer)
    buffer.seek(0)
    return buffer.read()

def bytes_to_package(byte_data):
    buffer = io.BytesIO(byte_data)
    return torch.load(buffer, weights_only=False)
# Kết thúc hiểu



# Bắt đầu hiểu
# HÀM NẠP CHECKPOINT THÔNG MINH BẰNG GLOB 
def load_checkpoints_safely():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    if NODE_ROLE == "node1":
        ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, "part1_*.pth"))
        if ckpts:
            latest_ckpt = max(ckpts, key=os.path.getmtime)
            part1_net.load_state_dict(torch.load(latest_ckpt, map_location="cpu"))
            print(f"[Node 1] - Successfully loaded part 1 checkpoint: {latest_ckpt}")
        else:
            print(f"[Node 1] - No part 1 checkpoint found, initializing new model from scratch.")

    elif NODE_ROLE == "node2":
        ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, "part2_*.pth"))
        if ckpts:
            latest_ckpt = max(ckpts, key=os.path.getmtime)
            part2_net.load_state_dict(torch.load(latest_ckpt, map_location="cpu"))
            print(f"[Node 2] - Successfully loaded part 2 checkpoint: {latest_ckpt}")
        else:
            print(f"[Node 2] - No part 2 checkpoint found, initializing new model from scratch.")

    elif NODE_ROLE == "node3":
        ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, "part3_*.pth"))
        if ckpts:
            latest_ckpt = max(ckpts, key=os.path.getmtime)
            part3_net.load_state_dict(torch.load(latest_ckpt, map_location="cpu"))
            print(f"[Node 3] - Successfully loaded part 3 checkpoint: {latest_ckpt}")
        else:
            print(f"[Node 3] - No part 3 checkpoint found, initializing new model from scratch.")
# Kết thúc hiểu




# =====================================================================
# XỬ LÝ CHO NODE 2 VÀ NODE 3 (FLASK SERVER NHẬN PIPELINE & BACKWARD)
# =====================================================================
# Bắt đầu hiểu
if NODE_ROLE in ["node2", "node3"]:
    load_checkpoints_safely()
    request_counter = 0
    interval_loss_sum = 0.0
    LOG_INTERVAL = 100  # Log every 200 requests
    counter_lock = threading.Lock()  # Lock for thread-safe counter increment
    @app.route("/process", methods=["POST"])
    def process_layer():
        global request_counter, interval_loss_sum
        start_time = time.time()
        try:

            file_data = request.files["package"].read()
            package = bytes_to_package(file_data)
            
            input_tensor = package["tensor"].clone().detach().requires_grad_(True)
            labels = package["label"]
            
            if NODE_ROLE == "node2":
                part2_net.train()
                output_tensor = part2_net(input_tensor)
                
                # Chuyển tiếp sang Node 3
                next_package = {"tensor": output_tensor, "label": labels}
                package_bytes = package_to_bytes(next_package)
                next_url = f"http://{IP_NODE_3}:5000/process"
                
                response = requests.post(next_url, files={"package": package_bytes}, timeout=30)
                resp_data = response.json()
# Kết thúc hiểu



                if resp_data.get("status") != "success":
                    return jsonify({"status": "error", "message": "Node 3 processing failed"}), 500
                
                grad_out2 = bytes_to_tensor(bytes.fromhex(resp_data["grad"]))
                
                # Backward & Optimizer cho Node 2
                optimizer2.zero_grad()
                output_tensor.backward(grad_out2)
                optimizer2.step()

                # Bắt đầu hiểu
                batch_loss = float(resp_data['loss'])
                # Lưu checkpoint liên tục
                torch.save(part2_net.state_dict(), os.path.join(CHECKPOINT_DIR, "part2_latest.pth"))
                elapsed = (time.time() - start_time) * 1000
                with counter_lock:
                    request_counter += 1
                    interval_loss_sum += batch_loss
                    current_count = request_counter
                # Trường hợp 1: Chạm mốc chu kỳ (ví dụ: 200, 400, 600...)
                if current_count % LOG_INTERVAL == 0:
                    avg_loss = interval_loss_sum / LOG_INTERVAL
                    print(f"[Node 2] Progress: {current_count} batches | Average Loss ({LOG_INTERVAL} steps): {avg_loss:.4f} | Last Batch Time: {elapsed:.2f}ms")
                    # Reset lại tổng tích lũy để tính cho chu kỳ tiếp theo
                    interval_loss_sum = 0.0
                # Trường hợp 2: In ngay batch đầu tiên để xác nhận server đã nhận request thành công
                elif current_count == 1:
                    print(f"[Node 2] Progress: {current_count} batches | Loss: {batch_loss:.4f} | First Batch Time: {elapsed:.2f}ms")
                    interval_loss_sum = 0.0  # Reset mốc bắt đầu
                # Kết thúc hiểu


                # Bắt đầu hiểu
                return jsonify({
                    "status": "success",
                    "grad": tensor_to_bytes(input_tensor.grad).hex(),
                    "correct": resp_data["correct"],
                    "total": resp_data["total"],
                    "loss": resp_data["loss"]
                }), 200
                # Kết thúc hiểu

# Bắt đầu hiểu
            elif NODE_ROLE == "node3":
                part3_net.train()
                outputs = part3_net(input_tensor)
                loss = criterion(outputs, labels)
# Kết thúc hiểu



                # Backward & Optimizer cho Node 3
                optimizer3.zero_grad()
                loss.backward()
                optimizer3.step()

                
                # Bắt đầu hiểu
                # Lưu checkpoint liên tục
                torch.save(part3_net.state_dict(), os.path.join(CHECKPOINT_DIR, "part3_latest.pth"))
                
                grad_out2 = input_tensor.grad
                
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels).sum().item()
                total = labels.size(0)
                
                batch_loss = float(loss.item())
                elapsed = (time.time() - start_time) * 1000
                # Kết thúc hiểu

                # Bắt đầu hiểu
                with counter_lock:
                    request_counter += 1
                    interval_loss_sum += batch_loss
                    current_count = request_counter     
                if current_count % LOG_INTERVAL == 0:
                    avg_loss = interval_loss_sum / LOG_INTERVAL
                    current_acc = (correct / total) * 100 # Hoặc tính trung bình accuracy nếu muốn
                    print(f"[Node 3] Progress: {current_count} batches | Average Loss ({LOG_INTERVAL} steps): {avg_loss:.4f} | Correct: {correct}/{total} | Time: {elapsed:.2f}ms")
                    interval_loss_sum = 0.0
                elif current_count == 1:
                    print(f"[Node 3] Progress: {current_count} batches | Loss: {batch_loss:.4f} | Correct: {correct}/{total} | First Batch Time: {elapsed:.2f}ms")
                    interval_loss_sum = 0.0
                
                return jsonify({
                    "status": "success",
                    "grad": tensor_to_bytes(grad_out2).hex(),
                    "correct": correct,
                    "total": total,
                    "loss": loss.item()
                }), 200
                
        except Exception as e:
            print(f"Error details: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
                # Kết thúc hiểu

# =====================================================================
# XỬ LÝ CHO NODE 1 (ĐIỀU PHỐI DỮ LIỆU, TRAIN & LOG KỸ THUẬT)
# =====================================================================
# Bắt đầu hiểu
def run_node_1():
    print(f"\n[Node 1] - Loading local CIFAR-10 dataset")
    load_checkpoints_safely()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
    
    TOTAL_EPOCHS = 1  # Số epoch Fine-tuning tối ưu trên thiết bị nhúng
    print(f"[Node 1] Training info:")
    print(f"         - Total batches / epoch: {len(trainloader)}")
    print(f"         - Batch size: 16")
    print(f"         - Total epochs to run : {TOTAL_EPOCHS}")
    print(f"[Node 1] Starting full-system fine-tuning")
    time.sleep(15) 
    for epoch in range(TOTAL_EPOCHS):
        epoch_start_time = time.time()
        print(f"==================================================")
        print(f"STARTING FINE-TUNE EPOCH {epoch+1}/{TOTAL_EPOCHS}")
        print(f"==================================================")
        
        total_correct = 0
        total_samples = 0
        running_loss = 0.0
# Kết thúc hiểu

        # bắt đầu hiểu
        for i, (images, labels) in enumerate(trainloader):
            batch_start = time.time()
            try:
                part1_net.train()
                input_tensor = images.clone().detach().requires_grad_(True)
                
                # 1. Forward Pass Phần 1 tại Node 1
                out_part1 = part1_net(input_tensor)
                
                # 2. Đóng gói và gửi sang Node 2 qua HTTP API
                package = {"tensor": out_part1, "label": labels}
                package_bytes = package_to_bytes(package)
                target_url = f"http://{IP_NODE_2}:5000/process"
                
                response = requests.post(target_url, files={"package": package_bytes}, timeout=30)
                resp_data = response.json()
        # kết thúc hiểu


                # Xử lý phản hồi và backward pass (tại node 1)
                if resp_data.get("status") == "success":
                    grad_out1 = bytes_to_tensor(bytes.fromhex(resp_data["grad"]))            
                    # 3. Backward Pass & Optimizer tại Node 1
                    optimizer1.zero_grad()
                    out_part1.backward(grad_out1)
                    optimizer1.step()
                    
                    # Thống kê thông số kỹ thuật
                    correct_batch = resp_data["correct"]
                    total_batch = resp_data["total"]
                    loss_batch = resp_data["loss"]
                    
                    total_correct += correct_batch
                    total_samples += total_batch
                    running_loss += loss_batch
                    
                    # In log chi tiết, chuyên nghiệp và ngắn gọn mỗi 100 batches
                    if (i + 1) % 100 == 0 or (i + 1) == len(trainloader):
                        avg_loss = running_loss / (i + 1)
                        current_acc = (total_correct / total_samples) * 100
                        batch_time = (time.time() - batch_start) * 1000
                        print(f"  [Node 1] Epoch [{epoch+1}/{TOTAL_EPOCHS}] | Batch [{i+1}/{len(trainloader)}] | "
                              f"Loss TB: {avg_loss:.4f} | Accuracy: {current_acc:.2f}% | Latency: {batch_time:.1f}ms")
                else:
                    print(f"  [Node 1 Warning] Lỗi phản hồi từ Node 2 ở batch {i+1}: {resp_data}")
                
            except Exception as e:
                print(f"  [Node 1 Connection Error] Batch {i+1}: {e}")
# Bắt đầu hiểu               
        # Tổng kết thông số kỹ thuật sau mỗi Epoch
        epoch_stop_time = time.time()
        epoch_duration = epoch_stop_time - epoch_start_time
        if total_samples > 0:
            epoch_acc = (total_correct / total_samples) * 100
        else:
            epoch_acc = 0.0
        num_batches = len(trainloader)
        if num_batches > 0:
            epoch_loss = running_loss / num_batches
        else:
            epoch_loss = 0.0
        
        print(f"\n--------------------------------------------------")
        print(f"EPOCH {epoch+1}/{TOTAL_EPOCHS} COMPLETED")
        print(f"Summary metrics:")
        print(f"   - Average epoch loss: {epoch_loss:.4f}")
        print(f"   - Accuracy: {epoch_acc:.2f}%")
        print(f"   - Total samples processed: {total_samples}")
        print(f"   - Epoch duration: {epoch_duration:.2f} seconds")
        print(f"----------------------------------------------------\n")
        
        # Lưu checkpoint Phần 1 sau khi kết thúc mỗi epoch
        ckpt_filename = os.path.join(CHECKPOINT_DIR, "part1_latest.pth")
        torch.save(part1_net.state_dict(), ckpt_filename)
        print(f"Safe checkpoint saved at: {ckpt_filename}\n")

    print(f"THE FINE-TUNING PROCESS HAS BEEN FULLY COMPLETED!")
# Kết thúc hiểu

if __name__ == "__main__":
    if NODE_ROLE == "node1":
        run_node_1()
    else:
        app.run(host="0.0.0.0", port=5000, debug=False)