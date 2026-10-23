import io
import os
import time
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from flask import Flask, jsonify, request

app = Flask(__name__)

# --- CẤU HÌNH ĐỊA CHỈ IP VÀ VAI TRÒ NODE ---
IP_NODE_2 = os.getenv("IP_NODE_2", "raspberry_pi_node_2")
IP_NODE_3 = os.getenv("IP_NODE_3", "raspberry_pi_node_3")
NODE_ROLE = os.getenv("NODE_ROLE", "node1")

# Khởi tạo mô hình MobileNetV3-Small với 10 lớp phân loại (CIFAR-10)
full_model = models.mobilenet_v3_small(weights=None, num_classes=10)
features = full_model.features

# Phân rã mô hình thành 3 phần tuần tự cho 3 Node
part1_net = nn.Sequential(*features[0:5])
part2_net = nn.Sequential(*features[5:10])
part3_net = nn.Sequential(
    *features[10:], full_model.avgpool, nn.Flatten(), full_model.classifier
)

# Khởi tạo Optimizer và Loss Function
optimizer1 = optim.Adam(part1_net.parameters(), lr=0.001)
optimizer2 = optim.Adam(part2_net.parameters(), lr=0.001)
optimizer3 = optim.Adam(part3_net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# --- HÀM HỖ TRỢ CHUYỂN ĐỔI DỮ LIỆU ---
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


# --- XỬ LÝ CHO NODE 2 VÀ NODE 3 (SERVER NHẬN PIPELINE & BACKWARD) ---
if NODE_ROLE in ["node2", "node3"]:
    # Tự động nạp checkpoint mới nhất cho Node 2 và Node 3 nếu có
    os.makedirs("./data/checkpoints", exist_ok=True)
    if NODE_ROLE == "node2":
        ckpt_path = "./data/checkpoints/part2_latest.pth"
        if os.path.exists(ckpt_path):
            part2_net.load_state_dict(torch.load(ckpt_path))
            print(f"[Node 2] Đã khôi phục trọng số từ checkpoint: {ckpt_path}")
    elif NODE_ROLE == "node3":
        ckpt_path = "./data/checkpoints/part3_latest.pth"
        if os.path.exists(ckpt_path):
            part3_net.load_state_dict(torch.load(ckpt_path))
            print(f"[Node 3] Đã khôi phục trọng số từ checkpoint: {ckpt_path}")

    @app.route("/process", methods=["POST"])
    def process_layer():
        try:
            file_data = request.files["package"].read()
            package = bytes_to_package(file_data)
            
            input_tensor = package["tensor"].clone().detach().requires_grad_(True)
            labels = package["label"]
            
            if NODE_ROLE == "node2":
                part2_net.train()
                output_tensor = part2_net(input_tensor)
                
                next_package = {"tensor": output_tensor, "label": labels}
                package_bytes = package_to_bytes(next_package)
                next_url = f"http://{IP_NODE_3}:5000/process"
                
                response = requests.post(next_url, files={"package": package_bytes}, timeout=30)
                resp_data = response.json()
                
                if resp_data.get("status") != "success":
                    return jsonify({"status": "error", "message": "Node 3 failed"}), 500
                
                grad_out2 = bytes_to_tensor(bytes.fromhex(resp_data["grad"]))
                
                optimizer2.zero_grad()
                output_tensor.backward(grad_out2)
                optimizer2.step()
                
                # Lưu checkpoint liên tục cho phần 2
                torch.save(part2_net.state_dict(), "./data/checkpoints/part2_latest.pth")
                
                grad_out1 = input_tensor.grad
                
                return jsonify({
                    "status": "success",
                    "grad": tensor_to_bytes(grad_out1).hex(),
                    "correct": resp_data["correct"],
                    "total": resp_data["total"],
                    "loss": resp_data["loss"]
                }), 200

            elif NODE_ROLE == "node3":
                part3_net.train()
                outputs = part3_net(input_tensor)
                loss = criterion(outputs, labels)
                
                optimizer3.zero_grad()
                loss.backward()
                optimizer3.step()
                
                # Lưu checkpoint liên tục cho phần 3
                torch.save(part3_net.state_dict(), "./data/checkpoints/part3_latest.pth")
                
                grad_out2 = input_tensor.grad
                
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels).sum().item()
                total = labels.size(0)
                
                return jsonify({
                    "status": "success",
                    "grad": tensor_to_bytes(grad_out2).hex(),
                    "correct": correct,
                    "total": total,
                    "loss": loss.item()
                }), 200

        except Exception as e:
            print(f"[{NODE_ROLE} Error] {e}")
            return jsonify({"status": "error", "message": str(e)}), 500


# --- XỬ LÝ CHO NODE 1 (TỰ ĐỘNG KHÔI PHỤC EPOCH ĐANG DỞ VÀ TRAINING TIẾP) ---
def run_node_1():
    print(f"[Node 1] Đang nạp dataset CIFAR-10 từ thư mục ./data ...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
    
    TOTAL_EPOCHS = 10
    start_epoch = 0
    
    # Kiểm tra thư mục checkpoint để tìm xem đã train đến epoch mấy
    checkpoints_dir = "./data/checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    existing_ckpts = [f for f in os.listdir(checkpoints_dir) if f.startswith("part1_epoch_")]
    if existing_ckpts:
        epochs = [int(f.split("_")[2].split(".")[0]) for f in existing_ckpts]
        latest_epoch = max(epochs)
        ckpt_file = os.path.join(checkpoints_dir, f"part1_epoch_{latest_epoch}.pth")
        
        part1_net.load_state_dict(torch.load(ckpt_file))
        start_epoch = latest_epoch  # Bắt đầu tiếp từ epoch sau epoch đã lưu
        print(f"\n[Node 1] 🟢 TÌM THẤY CHECKPOINT CŨ! Đã khôi phục từ Epoch {latest_epoch}.")
        print(f"[Node 1] Sẽ tiếp tục training từ Epoch {start_epoch + 1} đến {TOTAL_EPOCHS}...\n")
    else:
        print(f"\n[Node 1] Không tìm thấy checkpoint cũ, bắt đầu training mới từ đầu ({TOTAL_EPOCHS} Epochs)...\n")
        
    time.sleep(5) 
    
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        print(f"\n================ BẮT ĐẦU EPOCH {epoch+1}/{TOTAL_EPOCHS} ================")
        total_correct = 0
        total_samples = 0
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(trainloader):
            try:
                part1_net.train()
                input_tensor = images.clone().detach().requires_grad_(True)
                
                # 1. Forward Pass Phần 1
                out_part1 = part1_net(input_tensor)
                
                # 2. Gửi sang Node 2
                package = {"tensor": out_part1, "label": labels}
                package_bytes = package_to_bytes(package)
                target_url = f"http://{IP_NODE_2}:5000/process"
                
                response = requests.post(target_url, files={"package": package_bytes}, timeout=30)
                resp_data = response.json()
                
                if resp_data.get("status") == "success":
                    grad_out1 = bytes_to_tensor(bytes.fromhex(resp_data["grad"]))
                    
                    # 3. Backward Pass & Optimize Phần 1
                    optimizer1.zero_grad()
                    out_part1.backward(grad_out1)
                    optimizer1.step()
                    
                    total_correct += resp_data["correct"]
                    total_samples += resp_data["total"]
                    running_loss += resp_data["loss"]
                    
                    if (i + 1) % 50 == 0:
                        print(f"[Node 1] Epoch {epoch+1} | Batch {i+1}/{len(trainloader)} | Loss TB: {running_loss/(i+1):.4f} | Accuracy: {total_correct/total_samples*100:.2f}%")
                
            except Exception as e:
                print(f"[Node 1 Connection Error]: {e}")
                
        epoch_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0
        epoch_loss = running_loss / len(trainloader)
        
        print(f"\n---> HOÀN THÀNH EPOCH {epoch+1}/{TOTAL_EPOCHS}")
        print(f"---> Loss Epoch: {epoch_loss:.4f} | Accuracy Epoch: {epoch_acc:.2f}%\n")
        
        # Lưu checkpoint phần 1 sau khi hoàn thành epoch
        torch.save(part1_net.state_dict(), f"./data/checkpoints/part1_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    if NODE_ROLE == "node1":
        run_node_1()
    else:
        app.run(host="0.0.0.0", port=5000, debug=False)