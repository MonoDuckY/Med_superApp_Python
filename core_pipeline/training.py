import time
import requests
import logging
import os
import json
import zipfile
import tempfile
from .evaluate import evaluate_yolo_model

logger = logging.getLogger(__name__)

# Dictionary to store job statuses in memory
job_statuses = {}

def train_yolo_resnet(job_id: str, model_type: str, epochs: int, webhook_url: str, dataset_path: str = None):
    """
    Hàm giả lập quá trình Training YOLOv26 + ResNet50.
    Chạy ngầm trong BackgroundTask. Khi xong sẽ gọi evaluation và bắn webhook.
    """
    job_statuses[job_id] = {
        "status": "initializing",
        "progress": "Starting...",
        "metrics": None
    }
    
    logger.info(f"[JOB {job_id}] Bắt đầu quá trình Training {model_type} với {epochs} epochs...")
    if dataset_path:
        logger.info(f"[JOB {job_id}] Sử dụng dataset từ: {dataset_path}")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error(f"[JOB {job_id}] Thư viện ultralytics chưa được cài đặt. Vui lòng cài đặt qua pip.")
        job_statuses[job_id]["status"] = "failed"
        job_statuses[job_id]["progress"] = "Thiếu thư viện ultralytics."
        return

    job_statuses[job_id]["status"] = "training"
    
    # 1. Giải nén dataset vào thư mục tạm
    with tempfile.TemporaryDirectory() as temp_dir:
        extract_dir = os.path.join(temp_dir, "dataset")
        os.makedirs(extract_dir, exist_ok=True)
        
        if dataset_path and os.path.exists(dataset_path):
            job_statuses[job_id]["progress"] = "Đang giải nén dataset..."
            with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        else:
            job_statuses[job_id]["status"] = "failed"
            job_statuses[job_id]["progress"] = f"Không tìm thấy file dataset: {dataset_path}"
            return
            
        # Tìm file data.yaml trong thư mục giải nén
        yaml_path = None
        for root, dirs, files in os.walk(extract_dir):
            if "data.yaml" in files:
                yaml_path = os.path.join(root, "data.yaml")
                break
                
        if not yaml_path:
            job_statuses[job_id]["status"] = "failed"
            job_statuses[job_id]["progress"] = "Dataset không có file data.yaml hợp lệ."
            return
            
        # Tự động cấu hình lại đường dẫn tuyệt đối cho YOLO để tránh lỗi missing path
        try:
            import yaml
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data_yaml = yaml.safe_load(f)
            
            data_yaml['path'] = os.path.abspath(os.path.dirname(yaml_path))
            
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data_yaml, f)
        except Exception as e:
            logger.warning(f"Không thể cập nhật path trong data.yaml: {e}")

        # 2. Bắt đầu quá trình Training thực tế với Ultralytics YOLO
        job_statuses[job_id]["progress"] = f"Đang huấn luyện mô hình ({epochs} epochs)..."
        
        # Chọn model nền tương ứng với model_type
        base_model = "yolo26m.pt"  # Sử dụng YOLOv26 medium (file bạn vừa tải lên)
        if "yolov11" in model_type:
            base_model = "yolo11n.pt"
            
        try:
            model = YOLO(base_model)
            
            # Khởi chạy training
            # Use absolute path for project to prevent YOLO from saving in temp_dir or datasets_dir
            abs_weights_dir = os.path.abspath("weights")
            results = model.train(
                data=yaml_path,
                epochs=epochs,
                project=abs_weights_dir,
                name=f"trained_{model_type}_{job_id}",
                verbose=False
            )
        except Exception as e:
            logger.error(f"[JOB {job_id}] Quá trình training thất bại: {e}")
            job_statuses[job_id]["status"] = "failed"
            job_statuses[job_id]["progress"] = f"Lỗi Training: {e}"
            return

        # Generate results.xlsx from results.csv
        try:
            import pandas as pd
            csv_path = os.path.join(abs_weights_dir, f"trained_{model_type}_{job_id}", "results.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df.to_excel(os.path.join(abs_weights_dir, f"trained_{model_type}_{job_id}", "results.xlsx"), index=False)
        except Exception as e:
            logger.warning(f"Không thể tạo file results.xlsx: {e}")

        logger.info(f"[JOB {job_id}] Quá trình Training hoàn tất!")
        
        job_statuses[job_id]["status"] = "evaluating"
        job_statuses[job_id]["progress"] = "Đang đánh giá mô hình..."
        
        # Run evaluation
        os.makedirs("weights", exist_ok=True)
        # Đường dẫn weights thực tế được tạo ra bởi YOLO: weights/trained_{model_type}_{job_id}/weights/best.pt
        model_path = f"weights/trained_{model_type}_{job_id}/weights/best.pt"
        
        metrics = evaluate_yolo_model(model_path, yaml_path)
    
    job_statuses[job_id]["status"] = "completed"
    job_statuses[job_id]["progress"] = "Hoàn thành"
    job_statuses[job_id]["metrics"] = metrics
    
    # Kết quả giả định
    result = {
        "job_id": job_id,
        "status": "success",
        "message": "Training và Evaluation hoàn tất.",
        "model_path": model_path,
        "metrics": metrics
    }
    
    # Lưu kết quả training ra file JSON
    os.makedirs("outputs", exist_ok=True)
    result_path = f"outputs/result_{job_id}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    logger.info(f"[JOB {job_id}] Đã lưu kết quả tại: {result_path}")
    
    # Bắn webhook nếu có cấu hình
    if webhook_url:
        try:
            resp = requests.post(webhook_url, json=result, timeout=10)
            logger.info(f"[JOB {job_id}] Webhook triggered: {resp.status_code}")
        except Exception as e:
            logger.error(f"[JOB {job_id}] Lỗi khi gọi Webhook: {e}")
