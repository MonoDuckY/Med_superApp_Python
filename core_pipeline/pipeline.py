import cv2
import numpy as np
import base64
from typing import Dict, Any

from .preprocess import detect_safe_area, remove_text_and_callipers
from .enhance import apply_srad
from .segment import MedSAM_InferenceModel

# Khởi tạo model ở cấp độ module (Singleton pattern) để tái sử dụng
medsam_model = MedSAM_InferenceModel()

def run_ultrasound_pipeline(image_bytes: bytes, patient_id: str = "Unknown") -> Dict[str, Any]:
    """
    Chạy toàn bộ pipeline xử lý ảnh siêu âm từ đầu đến cuối.
    1. Preprocess (Safe Area + Text Removal)
    2. Enhance (SRAD)
    3. Segment (MedSAM)
    4. Format Output JSON
    """
    # Convert bytes to numpy array (OpenCV format)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Không thể decode ảnh gốc.")

    # Bước 1: Tiền xử lý
    image, bbox = detect_safe_area(image)
    image = remove_text_and_callipers(image)

    # Bước 2: Tăng cường chất lượng
    image = apply_srad(image, n_iter=10)

    # Bước 3: Phân vùng
    predictions = medsam_model.predict(image, bbox)

    # Bước 4: Encode ảnh kết quả sang Base64 để gửi về Frontend (tuỳ chọn)
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    h, w = image.shape[:2]

    # Cấu trúc JSON Output chuẩn
    result_data = {
        "info": {
            "tool": "Med_superApp AI Pipeline",
            "model": "MedSAM + SRAD + YOLO",
        },
        "image": {
            "width": w,
            "height": h,
            "patient_id": patient_id,
            "processed_image_base64": f"data:image/jpeg;base64,{img_base64}"
        },
        "annotations": predictions.get("annotations", [])
    }

    return result_data

import tempfile
import zipfile
import os
import requests
import logging

logger = logging.getLogger(__name__)

def batch_process_dataset(job_id: str, zip_bytes: bytes, webhook_url: str, options: dict):
    """
    Tiền xử lý hàng loạt ảnh từ file zip. Chạy ngầm trong BackgroundTask.
    Sau khi xử lý xong, nén lại và gửi thông báo qua Webhook.
    """
    logger.info(f"[JOB {job_id}] Bắt đầu xử lý dataset...")
    
    # Tạo thư mục tạm để làm việc
    with tempfile.TemporaryDirectory() as temp_dir:
        input_zip_path = os.path.join(temp_dir, "input.zip")
        extract_dir = os.path.join(temp_dir, "extracted")
        output_dir = os.path.join(temp_dir, "processed")
        output_zip_path = os.path.join(temp_dir, f"processed_{job_id}.zip")
        
        os.makedirs(extract_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Ghi file zip gốc
        with open(input_zip_path, 'wb') as f:
            f.write(zip_bytes)
            
        # Giải nén
        with zipfile.ZipFile(input_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        # Duyệt qua các ảnh và xử lý
        processed_count = 0
        for root, dirs, files in os.walk(extract_dir):
            for file_name in files:
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file_name)
                    image = cv2.imread(img_path)
                    
                    if image is None:
                        continue
                        
                    # Preprocess tùy theo options
                    if options.get("enable_safe_area", True):
                        image, _ = detect_safe_area(image)
                    if options.get("enable_text_removal", True):
                        image = remove_text_and_callipers(image)
                    if options.get("enable_srad", True):
                        n_iter = options.get("srad_iterations", 10)
                        image = apply_srad(image, n_iter=n_iter)
                        
                    # Lưu lại
                    out_path = os.path.join(output_dir, file_name)
                    cv2.imwrite(out_path, image)
                    processed_count += 1
                    
        # Nén lại thành file zip mới
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)
                    
        logger.info(f"[JOB {job_id}] Đã xử lý {processed_count} ảnh.")
        
        # Gửi Webhook báo cáo (Thực tế nên upload file zip này lên một Storage server local rồi gửi URL qua Webhook,
        # tạm thời gửi URL giả lập để Spring Boot biết là đã hoàn thành)
        result = {
            "job_id": job_id,
            "status": "success",
            "processed_count": processed_count,
            "download_url": f"http://127.0.0.1:8000/download/processed_{job_id}.zip" # URL giả lập
        }
        
        if webhook_url:
            try:
                requests.post(webhook_url, json=result, timeout=10)
                logger.info(f"[JOB {job_id}] Webhook called successfully.")
            except Exception as e:
                logger.error(f"[JOB {job_id}] Failed to call webhook: {e}")
