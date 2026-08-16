import time
import requests
import logging
import os
import json
import zipfile
import tempfile
import xml.etree.ElementTree as ET
import random
import shutil
from collections import defaultdict
import yaml

from .evaluate import evaluate_yolo_model

logger = logging.getLogger(__name__)

# Dictionary to store job statuses in memory
job_statuses = {}

def _xml_to_yolo_lines(xml_file_path: str, class_map: dict = None) -> list:
    """Chuyển đổi file Pascal VOC XML sang định dạng dòng nhãn YOLO."""
    if class_map is None:
        class_map = {'plus': 0, 'x_mark': 1, 'caliper': 2}
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        w_node = root.find('size/width')
        h_node = root.find('size/height')
        if w_node is None or h_node is None:
            return []
        w = float(w_node.text)
        h = float(h_node.text)
        if w <= 0 or h <= 0:
            return []
            
        lines = []
        for obj in root.findall('object'):
            name_node = obj.find('name')
            if name_node is None or not name_node.text:
                continue
            name = name_node.text.strip().lower()
            cls_id = class_map.get(name, 2)
            bnd = obj.find('bndbox')
            if bnd is None:
                continue
            xmin = float(bnd.find('xmin').text)
            ymin = float(bnd.find('ymin').text)
            xmax = float(bnd.find('xmax').text)
            ymax = float(bnd.find('ymax').text)
            
            x_center = ((xmin + xmax) / 2.0) / w
            y_center = ((ymin + ymax) / 2.0) / h
            box_w = (xmax - xmin) / w
            box_h = (ymax - ymin) / h
            
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            box_w = max(0.0, min(1.0, box_w))
            box_h = max(0.0, min(1.0, box_h))
            
            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
        return lines
    except Exception as e:
        logger.warning(f"Không thể parse XML {xml_file_path}: {e}")
        return []

def prepare_yolo_dataset(extract_dir: str) -> str:
    """
    Tự động chuẩn hóa dataset về format YOLO và trả về đường dẫn yaml_path.
    Hỗ trợ 2 trường hợp:
    1. Dataset đã có data.yaml: Cập nhật đường dẫn tuyệt đối path cho YOLO.
    2. Dataset dạng Pascal VOC XML / chưa có data.yaml:
       Tự động tìm ảnh & XML, convert sang YOLO .txt, chia Train/Val (85/15) và sinh data.yaml.
    """
    # 1. Tìm data.yaml
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f.lower() in ("data.yaml", "data.yml"):
                yaml_path = os.path.join(root, f)
                try:
                    with open(yaml_path, 'r', encoding='utf-8') as yf:
                        data_yaml = yaml.safe_load(yf) or {}
                    data_yaml['path'] = os.path.abspath(os.path.dirname(yaml_path))
                    with open(yaml_path, 'w', encoding='utf-8') as yf:
                        yaml.dump(data_yaml, yf)
                    return yaml_path
                except Exception as e:
                    logger.warning(f"Lỗi khi đọc/ghi data.yaml có sẵn: {e}")

    # 2. Nếu chưa có data.yaml, quét tìm file XML và ảnh
    xml_map = {}  # stem -> full_path
    img_map = {}  # stem -> full_path
    
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            stem, ext = os.path.splitext(f)
            ext_lower = ext.lower()
            if ext_lower == ".xml":
                xml_map[stem] = os.path.join(root, f)
            elif ext_lower in (".jpg", ".jpeg", ".png", ".bmp"):
                img_map[stem] = os.path.join(root, f)
                
    if not img_map:
        raise ValueError("Không tìm thấy tệp hình ảnh nào trong dataset.")
        
    logger.info(f"Tự động chuẩn hóa dataset VOC XML sang YOLO: Tìm thấy {len(img_map)} ảnh và {len(xml_map)} file XML.")
    
    # Tạo thư mục đích chuẩn YOLO
    yolo_dir = os.path.join(extract_dir, "yolo_dataset")
    train_img_dir = os.path.join(yolo_dir, "images", "train")
    val_img_dir = os.path.join(yolo_dir, "images", "val")
    train_lbl_dir = os.path.join(yolo_dir, "labels", "train")
    val_lbl_dir = os.path.join(yolo_dir, "labels", "val")
    
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)
        
    # Gom nhóm theo base ID để tránh rò rỉ dữ liệu tăng cường
    groups = defaultdict(list)
    for stem, img_path in img_map.items():
        base_id = stem.replace('_aug_synthetic_caliper', '').replace('_aug_flip_h', '').replace('_aug_flip_v', '').replace('_aug_rot_180', '')
        groups[base_id].append((stem, img_path))
        
    base_ids = sorted(list(groups.keys()))
    random.seed(42)
    random.shuffle(base_ids)
    
    train_count = max(1, int(len(base_ids) * 0.85))
    train_base_ids = set(base_ids[:train_count])
    
    for base_id, items in groups.items():
        is_train = base_id in train_base_ids
        target_img_dir = train_img_dir if is_train else val_img_dir
        target_lbl_dir = train_lbl_dir if is_train else val_lbl_dir
        
        for stem, src_img_path in items:
            dst_img_path = os.path.join(target_img_dir, f"{stem}.jpg")
            shutil.copy2(src_img_path, dst_img_path)
            
            # Xử lý nhãn
            txt_lines = []
            if stem in xml_map:
                txt_lines = _xml_to_yolo_lines(xml_map[stem])
                
            dst_lbl_path = os.path.join(target_lbl_dir, f"{stem}.txt")
            with open(dst_lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(txt_lines))
                
    # Sinh data.yaml
    data_yaml_path = os.path.join(yolo_dir, "data.yaml")
    data_yaml_content = {
        "path": os.path.abspath(yolo_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": 3,
        "names": ["plus", "x_mark", "caliper"]
    }
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_content, f)
        
    logger.info(f"Đã tự động tạo data.yaml thành công tại: {data_yaml_path}")
    return data_yaml_path

def train_yolo_resnet(job_id: str, model_type: str, epochs: int, webhook_url: str, dataset_path: str = None):
    """
    Hàm thực hiện quá trình Training YOLO.
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
            
        # Chuẩn bị dataset & lấy đường dẫn data.yaml
        job_statuses[job_id]["progress"] = "Đang kiểm tra và chuẩn bị cấu trúc dataset..."
        try:
            yaml_path = prepare_yolo_dataset(extract_dir)
        except Exception as e:
            logger.error(f"[JOB {job_id}] Lỗi chuẩn bị dataset: {e}")
            job_statuses[job_id]["status"] = "failed"
            job_statuses[job_id]["progress"] = f"Dataset không hợp lệ: {e}"
            return

        # 2. Bắt đầu quá trình Training thực tế với Ultralytics YOLO
        job_statuses[job_id]["progress"] = f"Đang huấn luyện mô hình ({epochs} epochs)..."
        
        # Chọn model nền tương ứng với model_type
        base_model = "yolov8n.pt"  # Mặc định sử dụng YOLOv8 nano cho nhanh
        if "yolov11" in model_type or "yolo11" in model_type:
            base_model = "yolo11n.pt"
            
        try:
            model = YOLO(base_model)
            
            # Khởi chạy training
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
    
    # Kết quả
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
