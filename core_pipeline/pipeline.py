import cv2
import numpy as np
import base64
import os
import tempfile
import zipfile
import requests
import logging
import threading
from typing import Dict, Any

from .preprocess import detect_safe_area, remove_text_and_callipers, detect_calipers, highlight_calipers_on_image
from .enhance import apply_srad, adjust_brightness_contrast, adjust_sharpness
from .xml_exporter import save_to_combined_xml, generate_xml_string
from .augment import augment_image_and_xml

logger = logging.getLogger(__name__)

# Thư mục chứa template caliper tĩnh (nằm cùng cấp với pipeline.py)
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

def send_progress_webhook(url: str, job_id: str, processed: int, total: int):
    """Gửi cập nhật tiến độ bất đồng bộ qua webhook."""
    if not url:
        return
    def _send():
        try:
            requests.post(
                url,
                json={"job_id": job_id, "status": "processing", "processed": processed, "total": total},
                timeout=3
            )
        except Exception as e:
            logger.debug(f"Progress webhook failed: {e}")
    threading.Thread(target=_send, daemon=True).start()

def process_single_image(image_bytes: bytes, filename: str = "ultrasound.jpg", options: dict = None) -> Dict[str, Any]:
    """
    Xử lý đơn ảnh siêu âm phục vụ Single Image Studio (UC21 & UC22 Preview).
    1. Cân chỉnh Brightness / Contrast / Sharpness
    2. Khử nhiễu SRAD (nếu bật)
    3. Cắt Safe Area (nếu bật)
    4. Nhận diện Calipers (+, x) bằng Template Matching
    5. Xóa chữ trên ảnh (nếu bật)
    6. Tô đỏ (Highlight) Calipers trên ảnh
    7. Sinh XML Pascal VOC và trả về Base64 cùng toạ độ JSON.
    """
    if options is None:
        options = {}

    nparr = np.frombuffer(image_bytes, np.uint8)
    orig_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if orig_img is None:
        raise ValueError("Không thể giải mã dữ liệu hình ảnh tải lên.")

    image = orig_img.copy()

    # 1. Cân chỉnh Brightness, Contrast, Sharpness
    brightness = int(options.get("brightness", 0))
    contrast = float(options.get("contrast", 1.0))
    sharpness = float(options.get("sharpness", 0.0))
    image = adjust_brightness_contrast(image, brightness, contrast)
    image = adjust_sharpness(image, sharpness)

    # 2. Khử nhiễu SRAD
    if options.get("enable_srad", True):
        srad_iter = int(options.get("srad_iterations", 10))
        if srad_iter > 0:
            image = apply_srad(image, n_iter=srad_iter)

    # 3. Lọc vùng quạt siêu âm (Safe Area)
    safe_bbox = [0, 0, image.shape[1], image.shape[0]]
    if options.get("enable_safe_area", False):
        image, safe_bbox = detect_safe_area(image)

    # 4. Phát hiện Caliper
    threshold = float(options.get("threshold", 0.62))
    caliper_mask, boxes = detect_calipers(image, TEMPLATES_DIR, threshold=threshold)

    # 5. Xóa chữ trên ảnh (Inpainting)
    if options.get("enable_text_removal", False):
        image = remove_text_and_callipers(image)

    # 6. Tô đỏ Caliper trên ảnh kết quả
    if options.get("highlight_caliper", True) and boxes:
        image = highlight_calipers_on_image(image, caliper_mask, boxes, color=(0, 0, 255), draw_box=True)

    # 7. Sinh chuỗi Pascal VOC XML
    xml_str = generate_xml_string(filename, image.shape, boxes)

    # 8. Encode ảnh gốc và ảnh kết quả sang Base64
    _, orig_buf = cv2.imencode('.jpg', orig_img)
    orig_base64 = base64.b64encode(orig_buf).decode('utf-8')

    _, proc_buf = cv2.imencode('.jpg', image)
    proc_base64 = base64.b64encode(proc_buf).decode('utf-8')

    h, w = image.shape[:2]

    return {
        "filename": filename,
        "width": w,
        "height": h,
        "original_image_base64": f"data:image/jpeg;base64,{orig_base64}",
        "processed_image_base64": f"data:image/jpeg;base64,{proc_base64}",
        "calipers": boxes,
        "caliper_count": len(boxes),
        "xml_content": xml_str,
        "options_applied": {
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
            "enable_srad": options.get("enable_srad", True),
            "srad_iterations": options.get("srad_iterations", 10),
            "enable_safe_area": options.get("enable_safe_area", False),
            "enable_text_removal": options.get("enable_text_removal", False),
            "highlight_caliper": options.get("highlight_caliper", True),
            "threshold": threshold
        }
    }

def run_ultrasound_pipeline(image_bytes: bytes, patient_id: str = "Unknown") -> Dict[str, Any]:
    """Hàm tương thích ngược với API cũ."""
    result = process_single_image(image_bytes, filename=f"patient_{patient_id}.jpg")
    return {
        "info": {
            "tool": "Med_superApp AI Pipeline",
            "model": "Caliper Detection & SRAD Filter",
        },
        "image": {
            "width": result["width"],
            "height": result["height"],
            "patient_id": patient_id,
            "processed_image_base64": result["processed_image_base64"]
        },
        "annotations": result["calipers"],
        "xml_content": result["xml_content"]
    }

def batch_process_dataset(job_id: str, zip_bytes: bytes, webhook_url: str, options: dict):
    """
    Tiền xử lý hàng loạt ảnh từ file zip theo cấu hình options (UC21 & UC22).
    Chạy ngầm trong BackgroundTasks, nén kết quả và gửi thông báo qua Webhook.
    """
    logger.info(f"[JOB {job_id}] Bắt đầu xử lý dataset...")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    output_zip_path = os.path.join(outputs_dir, f"processed_{job_id}.zip")

    with tempfile.TemporaryDirectory() as temp_dir:
        input_zip_path = os.path.join(temp_dir, "input.zip")
        extract_dir = os.path.join(temp_dir, "extracted")
        output_dir = os.path.join(temp_dir, "processed")

        images_dir = os.path.join(output_dir, "images")
        annotations_dir = os.path.join(output_dir, "annotations")
        os.makedirs(extract_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)

        with open(input_zip_path, 'wb') as f:
            f.write(zip_bytes)

        with zipfile.ZipFile(input_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Lấy danh sách ảnh hợp lệ
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        image_files = []
        for root, _, files in os.walk(extract_dir):
            for f in files:
                if f.lower().endswith(valid_extensions):
                    image_files.append(os.path.join(root, f))

        total_files = len(image_files)
        if total_files == 0:
            total_files = 1
        logger.info(f"[JOB {job_id}] Tìm thấy {len(image_files)} ảnh cần xử lý.")

        processed_count = 0
        total_calipers_detected = 0

        for idx, img_path in enumerate(image_files, start=1):
            file_name = os.path.basename(img_path)
            image = cv2.imread(img_path)
            if image is None:
                continue

            # 1. Điều chỉnh ảnh (Brightness, Contrast, Sharpness)
            brightness = int(options.get("brightness", 0))
            contrast = float(options.get("contrast", 1.0))
            sharpness = float(options.get("sharpness", 0.0))
            image = adjust_brightness_contrast(image, brightness, contrast)
            image = adjust_sharpness(image, sharpness)

            # 2. Safe Area
            if options.get("enable_safe_area", False):
                image, _ = detect_safe_area(image)

            # 3. Khử nhiễu SRAD
            if options.get("enable_srad", True):
                n_iter = int(options.get("srad_iterations", 10))
                if n_iter > 0:
                    image = apply_srad(image, n_iter=n_iter)

            # 4. Phát hiện Caliper
            threshold = float(options.get("threshold", 0.62))
            caliper_mask, boxes = detect_calipers(image, TEMPLATES_DIR, threshold=threshold)
            total_calipers_detected += len(boxes)

            # 5. Xóa chữ
            if options.get("enable_text_removal", False):
                image = remove_text_and_callipers(image)

            # 6. Tô đỏ Caliper
            if options.get("highlight_caliper", True) and boxes:
                image = highlight_calipers_on_image(image, caliper_mask, boxes, color=(0, 0, 255), draw_box=True)

            # 7. Lưu ảnh và file XML
            base_name = os.path.splitext(file_name)[0]
            out_img_name = f"{base_name}.jpg"
            cv2.imwrite(os.path.join(images_dir, out_img_name), image)
            save_to_combined_xml(
                os.path.join(annotations_dir, f"{base_name}.xml"),
                out_img_name,
                image.shape,
                boxes,
                folder_name="images"
            )
            processed_count += 1

            # 8. Augmentation x4 (nếu bật)
            if options.get("enable_augmentation", False):
                aug_results = augment_image_and_xml(image, boxes)
                for aug_img, aug_boxes, suffix in aug_results:
                    aug_name = f"{base_name}_{suffix}"
                    cv2.imwrite(os.path.join(images_dir, f"{aug_name}.jpg"), aug_img)
                    save_to_combined_xml(
                        os.path.join(annotations_dir, f"{aug_name}.xml"),
                        f"{aug_name}.jpg",
                        aug_img.shape,
                        aug_boxes,
                        folder_name="images"
                    )
                    processed_count += 1

            # Báo tiến độ định kỳ
            if idx % 5 == 0 or idx == len(image_files):
                logger.info(f"[JOB {job_id}] Tiến độ: {idx}/{len(image_files)} ảnh ({(idx/len(image_files)*100):.1f}%)")
                send_progress_webhook(webhook_url, job_id, idx, len(image_files))

        # Đóng gói zip
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(output_dir):
                for f in files:
                    full_p = os.path.join(root, f)
                    arc_p = os.path.relpath(full_p, output_dir)
                    zipf.write(full_p, arc_p)

        logger.info(f"[JOB {job_id}] Đã xử lý xong. Tạo ra {processed_count} files.")

        result = {
            "job_id": job_id,
            "status": "success",
            "processed_count": processed_count,
            "calipers_detected": total_calipers_detected,
            "download_url": f"http://127.0.0.1:8000/download/processed_{job_id}.zip"
        }

        if webhook_url:
            try:
                requests.post(webhook_url, json=result, timeout=10)
                logger.info(f"[JOB {job_id}] Webhook thông báo thành công.")
            except Exception as e:
                logger.error(f"[JOB {job_id}] Webhook thất bại: {e}")

