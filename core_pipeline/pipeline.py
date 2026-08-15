import cv2
import numpy as np
import base64
from typing import Dict, Any

from .preprocess import detect_safe_area, remove_text_and_callipers
from .enhance import apply_srad
# Tạm thời comment phần import AI model vì theo kế hoạch sẽ train AI sau
# from .segment import MedSAM_InferenceModel
class MedSAM_InferenceModel:
    def predict(self, image, bbox):
        return {"annotations": []}

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

from .preprocess import detect_safe_area, remove_text_and_callipers, detect_calipers
from .enhance import apply_srad, adjust_brightness_contrast, adjust_sharpness
from .xml_exporter import save_to_combined_xml
from .yolo_exporter import save_to_yolo_txt
from .augment import augment_image_and_xml

logger = logging.getLogger(__name__)

import threading
def send_progress_webhook(url, job_id, processed, total):
    if not url: return
    def _send():
        try:
            requests.post(url, json={"job_id": job_id, "status": "processing", "processed": processed, "total": total}, timeout=3)
        except Exception:
            pass
    threading.Thread(target=_send, daemon=True).start()

# Thư mục chứa template caliper tĩnh (nằm cùng cấp với pipeline.py)
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

def batch_process_dataset(job_id: str, zip_bytes: bytes, webhook_url: str, options: dict):
    """
    Tiền xử lý hàng loạt ảnh từ file zip theo cấu hình options (UC-23).
    Chạy ngầm trong BackgroundTask. Sau khi xử lý xong, nén lại và gửi thông báo qua Webhook.
    """
    logger.info(f"[JOB {job_id}] Bắt đầu xử lý dataset...")
    
    # Tạo thư mục outputs vĩnh viễn ở root project
    base_dir = os.path.dirname(os.path.dirname(__file__))
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        input_zip_path = os.path.join(temp_dir, "input.zip")
        extract_dir = os.path.join(temp_dir, "extracted")
        output_dir = os.path.join(temp_dir, "processed")
        
        # Đường dẫn tới file ZIP vĩnh viễn
        output_zip_path = os.path.join(outputs_dir, f"processed_{job_id}.zip")
        
        os.makedirs(extract_dir, exist_ok=True)
        
        # Tạo cấu trúc thư mục chuẩn YOLO
        images_dir = os.path.join(output_dir, "images", "train")
        labels_dir = os.path.join(output_dir, "labels", "train")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Tạo file data.yaml
        data_yaml_content = f"train: images/train\nval: images/train\n\nnc: 3\nnames: ['plus', 'x_mark', 'caliper']\n"
        with open(os.path.join(output_dir, "data.yaml"), "w") as f:
            f.write(data_yaml_content)
        
        with open(input_zip_path, 'wb') as f:
            f.write(zip_bytes)
            
        with zipfile.ZipFile(input_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        processed_count = 0
        current_file_index = 0
        
        # Đếm tổng số file ảnh để tính % tiến độ
        total_files = sum(1 for r, _, fs in os.walk(extract_dir) for f in fs if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        if total_files == 0:
            total_files = 1 # Tránh chia cho 0
        logger.info(f"[JOB {job_id}] Tìm thấy tổng cộng {total_files} ảnh gốc cần xử lý.")
        
        for root, dirs, files in os.walk(extract_dir):
            for file_name in files:
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    current_file_index += 1
                    img_path = os.path.join(root, file_name)
                    image = cv2.imread(img_path)
                    
                    if image is None:
                        continue
                        
                    # 1. Điều chỉnh ảnh (Brightness, Contrast, Sharpness)
                    brightness = options.get("brightness", 0)
                    contrast = options.get("contrast", 1.0)
                    sharpness = options.get("sharpness", 0.0)
                    image = adjust_brightness_contrast(image, brightness, contrast)
                    image = adjust_sharpness(image, sharpness)
                    
                    if options.get("enable_safe_area", False):
                        image, _ = detect_safe_area(image)
                        
                    if options.get("enable_srad", False):
                        n_iter = options.get("srad_iterations", 10)
                        image = apply_srad(image, n_iter=n_iter)
                        
                    # 2. Phát hiện Caliper và Trích xuất XML
                    caliper_mask, boxes = detect_calipers(image, TEMPLATES_DIR)
                    
                    # 3. Xóa Caliper & Chữ trên ảnh gốc (nếu được yêu cầu)
                    if options.get("enable_text_removal", False):
                        # Xóa caliper bằng mask vừa tìm được
                        image = cv2.inpaint(image, caliper_mask, 3, cv2.INPAINT_TELEA)
                        # Dùng OCR để xóa các chữ viết khác
                        image = remove_text_and_callipers(image)
                        
                    # 4. Lưu ảnh gốc đã xử lý và file YOLO txt của nó
                    base_name = os.path.splitext(file_name)[0]
                    cv2.imwrite(os.path.join(images_dir, f"{base_name}.jpg"), image)
                    save_to_yolo_txt(os.path.join(labels_dir, f"{base_name}.txt"), image.shape, boxes)
                    
                    # (Tùy chọn) Lưu thêm XML cho mục đích dự phòng, hoặc bỏ qua.
                    # save_to_combined_xml(os.path.join(output_dir, "annotations", f"{base_name}.xml"), f"{base_name}.jpg", image.shape, boxes)
                    
                    processed_count += 1
                    
                    # 5. Làm giàu dữ liệu (Augmentation - Optional)
                    if options.get("enable_augmentation", False):
                        aug_results = augment_image_and_xml(image, boxes)
                        for aug_img, aug_boxes, suffix in aug_results:
                            aug_name = f"{base_name}_{suffix}"
                            cv2.imwrite(os.path.join(images_dir, f"{aug_name}.jpg"), aug_img)
                            save_to_yolo_txt(os.path.join(labels_dir, f"{aug_name}.txt"), aug_img.shape, aug_boxes)
                            processed_count += 1
                    
                    # Báo cáo tiến độ (mỗi 5 file hoặc file cuối cùng)
                    if current_file_index % 5 == 0 or current_file_index == total_files:
                        logger.info(f"[JOB {job_id}] Đang xử lý: {current_file_index}/{total_files} ảnh ({(current_file_index/total_files*100):.1f}%)")
                        send_progress_webhook(webhook_url, job_id, current_file_index, total_files)
                    
        # 6. Đóng gói lại thành Zip
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)
                    
        logger.info(f"[JOB {job_id}] Đã xử lý và tạo ra {processed_count} files (cả Ảnh và XML).")
        
        result = {
            "job_id": job_id,
            "status": "success",
            "processed_count": processed_count,
            "download_url": f"http://127.0.0.1:8000/download/processed_{job_id}.zip"
        }
        
        if webhook_url:
            try:
                requests.post(webhook_url, json=result, timeout=10)
                logger.info(f"[JOB {job_id}] Webhook called successfully.")
            except Exception as e:
                logger.error(f"[JOB {job_id}] Failed to call webhook: {e}")
