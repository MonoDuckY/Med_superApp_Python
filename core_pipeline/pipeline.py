import cv2
import numpy as np
import base64
import json
import os
import tempfile
import zipfile
import requests
import logging
import threading
from typing import Dict, Any

from .preprocess import (
    detect_safe_area,
    remove_text_and_callipers,
    detect_calipers,
    highlight_calipers_on_image,
    remove_calipers
)
from .enhance import apply_srad, adjust_brightness_contrast, adjust_sharpness
from .xml_exporter import save_to_combined_xml, generate_xml_string
from .augment import augment_image_and_xml, draw_synthetic_caliper_pair, augment_with_synthetic_calipers

logger = logging.getLogger(__name__)

# Thư mục chứa template caliper tĩnh (nằm cùng cấp với pipeline.py)
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

def send_progress_webhook(url: str, job_id: str, processed: int, total: int, current_file: str = "", calipers_detected: int = 0):
    """Gửi cập nhật tiến độ chi tiết bất đồng bộ qua webhook."""
    if not url:
        return
    def _send():
        try:
            percent = round((processed / max(1, total)) * 100, 1)
            requests.post(
                url,
                json={
                    "job_id": job_id,
                    "status": "processing",
                    "processed": int(processed),
                    "total": int(total),
                    "percent": float(percent),
                    "current_file": str(current_file),
                    "calipers_detected": int(calipers_detected)
                },
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
    5. Xóa Calipers bảo tồn mô (nếu bật)
    6. Xóa chữ trên ảnh (nếu bật)
    7. Tô đỏ (Highlight) Calipers trên ảnh (nếu không chọn xóa)
    8. Sinh Caliper giả lập chống Shortcut Learning (nếu bật)
    9. Sinh XML Pascal VOC và trả về Base64 cùng toạ độ JSON.
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

    # 5. Xóa Caliper bảo tồn mô (Speckle-Aware Inpainting)
    caliper_removal_enabled = options.get("enable_caliper_removal", False)
    caliper_removal_method = options.get("caliper_removal_method", "speckle_aware")
    if caliper_removal_enabled and np.any(caliper_mask > 0):
        image = remove_calipers(image, caliper_mask, method=caliper_removal_method)

    # 6. Xóa chữ trên ảnh (Inpainting)
    if options.get("enable_text_removal", False):
        image = remove_text_and_callipers(image, method=caliper_removal_method)

    # 7. Tô đỏ Caliper trên ảnh kết quả (nếu không bật xóa Caliper)
    if not caliper_removal_enabled and options.get("highlight_caliper", True) and boxes:
        image = highlight_calipers_on_image(image, caliper_mask, boxes, color=(0, 0, 255), draw_box=True)

    # 8. Sinh Caliper giả lập chống Shortcut Learning (nếu bật)
    if options.get("enable_synthetic_caliper", False):
        image, _, syn_boxes = draw_synthetic_caliper_pair(image, safe_bbox=safe_bbox)
        boxes = list(boxes) + syn_boxes

    # 9. Sinh chuỗi Pascal VOC XML
    xml_str = generate_xml_string(filename, image.shape, boxes)

    # 10. Encode ảnh gốc và ảnh kết quả sang Base64
    _, orig_buf = cv2.imencode('.jpg', orig_img)
    orig_base64 = base64.b64encode(orig_buf).decode('utf-8')

    _, proc_buf = cv2.imencode('.jpg', image)
    proc_base64 = base64.b64encode(proc_buf).decode('utf-8')

    h, w = image.shape[:2]

    clean_boxes = [
        {
            "name": str(b.get("name", "caliper")),
            "xmin": int(b["xmin"]),
            "ymin": int(b["ymin"]),
            "xmax": int(b["xmax"]),
            "ymax": int(b["ymax"])
        }
        for b in boxes
    ]

    return {
        "filename": str(filename),
        "width": int(w),
        "height": int(h),
        "original_image_base64": f"data:image/jpeg;base64,{orig_base64}",
        "processed_image_base64": f"data:image/jpeg;base64,{proc_base64}",
        "calipers": clean_boxes,
        "caliper_count": int(len(clean_boxes)),
        "xml_content": str(xml_str),
        "options_applied": {
            "brightness": int(brightness),
            "contrast": float(contrast),
            "sharpness": float(sharpness),
            "enable_srad": bool(options.get("enable_srad", True)),
            "srad_iterations": int(options.get("srad_iterations", 10)),
            "enable_safe_area": bool(options.get("enable_safe_area", False)),
            "enable_text_removal": bool(options.get("enable_text_removal", False)),
            "enable_caliper_removal": bool(caliper_removal_enabled),
            "caliper_removal_method": str(caliper_removal_method),
            "enable_synthetic_caliper": bool(options.get("enable_synthetic_caliper", False)),
            "highlight_caliper": bool(options.get("highlight_caliper", True)),
            "threshold": float(threshold)
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

        # 3 Thư mục chính theo đúng quy chuẩn phân loại
        orig_images_dir = os.path.join(output_dir, "original_images")
        proc_images_dir = os.path.join(output_dir, "processed_images")
        coordinates_dir = os.path.join(output_dir, "coordinates")

        os.makedirs(extract_dir, exist_ok=True)
        os.makedirs(orig_images_dir, exist_ok=True)
        os.makedirs(proc_images_dir, exist_ok=True)
        os.makedirs(coordinates_dir, exist_ok=True)

        # Định dạng xuất toạ độ tùy chọn
        export_xml = options.get("export_xml", True)
        export_json = options.get("export_json", True)
        export_csv = options.get("export_csv", True)
        include_original = options.get("include_original", True)

        xml_dir = os.path.join(coordinates_dir, "xml")
        json_dir = os.path.join(coordinates_dir, "json")
        if export_xml:
            os.makedirs(xml_dir, exist_ok=True)
        if export_json:
            os.makedirs(json_dir, exist_ok=True)

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
        master_csv_rows = ["filename,caliper_id,type,xmin,ymin,xmax,ymax,center_x,center_y,width,height"]
        master_json_items = []

        for idx, img_path in enumerate(image_files, start=1):
            file_name = os.path.basename(img_path)
            orig_image = cv2.imread(img_path)
            if orig_image is None:
                continue

            base_name = os.path.splitext(file_name)[0]
            out_img_name = f"{base_name}.jpg"

            # 0. Lưu ảnh gốc vào thư mục original_images/
            if include_original:
                cv2.imwrite(os.path.join(orig_images_dir, out_img_name), orig_image)

            image = orig_image.copy()

            # 1. Điều chỉnh ảnh (nếu có cấu hình tùy chọn)
            brightness = int(options.get("brightness", 0))
            contrast = float(options.get("contrast", 1.0))
            sharpness = float(options.get("sharpness", 0.0))
            if brightness != 0 or contrast != 1.0:
                image = adjust_brightness_contrast(image, brightness, contrast)
            if sharpness > 0:
                image = adjust_sharpness(image, sharpness)

            # 2. Safe Area (nếu bật)
            if options.get("enable_safe_area", False):
                image, _ = detect_safe_area(image)

            # 3. Khử nhiễu SRAD (nếu bật)
            if options.get("enable_srad", False):
                n_iter = int(options.get("srad_iterations", 10))
                if n_iter > 0:
                    image = apply_srad(image, n_iter=n_iter)

            # 4. Phát hiện Caliper Đa tỉ lệ (Multi-Scale Template Matching)
            threshold = float(options.get("threshold", 0.58))
            caliper_mask, boxes = detect_calipers(image, TEMPLATES_DIR, threshold=threshold)
            total_calipers_detected += len(boxes)

            # 5. Xóa Caliper bảo tồn mô (Mặc định bật, hoặc theo options)
            caliper_removal_enabled = options.get("enable_caliper_removal", True)
            caliper_removal_method = options.get("caliper_removal_method", "speckle_aware")
            if caliper_removal_enabled and np.any(caliper_mask > 0):
                image = remove_calipers(image, caliper_mask, method=caliper_removal_method)

            # 6. Xóa chữ OCR (nếu bật)
            if options.get("enable_text_removal", False):
                image = remove_text_and_callipers(image, method=caliper_removal_method)

            # 7. Tô đỏ Caliper (nếu không bật xóa Caliper)
            if not caliper_removal_enabled and options.get("highlight_caliper", False) and boxes:
                image = highlight_calipers_on_image(image, caliper_mask, boxes, color=(0, 0, 255), draw_box=True)

            # 8. Lưu ảnh đã làm sạch vào processed_images/
            cv2.imwrite(os.path.join(proc_images_dir, out_img_name), image)
            processed_count += 1

            # Xuất XML nếu được chọn
            if export_xml:
                save_to_combined_xml(
                    os.path.join(xml_dir, f"{base_name}.xml"),
                    out_img_name,
                    image.shape,
                    boxes,
                    folder_name="processed_images"
                )

            # Chuẩn bị dữ liệu toạ độ từng ảnh
            img_item = {
                "filename": out_img_name,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "calipers_count": len(boxes),
                "calipers": []
            }
            for c_idx, b in enumerate(boxes, start=1):
                cx = int((b['xmin'] + b['xmax']) // 2)
                cy = int((b['ymin'] + b['ymax']) // 2)
                cw = int(b['xmax'] - b['xmin'])
                ch = int(b['ymax'] - b['ymin'])
                master_csv_rows.append(
                    f'"{out_img_name}",{c_idx},"{b["name"]}",{b["xmin"]},{b["ymin"]},{b["xmax"]},{b["ymax"]},{cx},{cy},{cw},{ch}'
                )
                img_item["calipers"].append({
                    "id": c_idx,
                    "type": str(b["name"]),
                    "xmin": int(b["xmin"]),
                    "ymin": int(b["ymin"]),
                    "xmax": int(b["xmax"]),
                    "ymax": int(b["ymax"]),
                    "center": [cx, cy],
                    "size": [cw, ch]
                })
            master_json_items.append(img_item)

            # Xuất file JSON đơn lẻ cho từng ảnh nếu được chọn
            if export_json:
                with open(os.path.join(json_dir, f"{base_name}.json"), "w", encoding="utf-8") as f_single_json:
                    json.dump(img_item, f_single_json, indent=2, ensure_ascii=False)

            # 9. Augmentation x4 hình học (nếu bật)
            if options.get("enable_augmentation", False):
                aug_results = augment_image_and_xml(image, boxes)
                for aug_img, aug_boxes, suffix in aug_results:
                    aug_name = f"{base_name}_{suffix}"
                    cv2.imwrite(os.path.join(proc_images_dir, f"{aug_name}.jpg"), aug_img)
                    if export_xml:
                        save_to_combined_xml(
                            os.path.join(xml_dir, f"{aug_name}.xml"),
                            f"{aug_name}.jpg",
                            aug_img.shape,
                            aug_boxes,
                            folder_name="processed_images"
                        )
                    processed_count += 1

            # 10. Synthetic Caliper Augmentation (nếu bật - chống Shortcut Learning)
            if options.get("enable_synthetic_calipers", False):
                syn_results = augment_with_synthetic_calipers(image, boxes)
                for syn_img, syn_boxes, suffix in syn_results:
                    syn_name = f"{base_name}_{suffix}"
                    cv2.imwrite(os.path.join(proc_images_dir, f"{syn_name}.jpg"), syn_img)
                    if export_xml:
                        save_to_combined_xml(
                            os.path.join(xml_dir, f"{syn_name}.xml"),
                            f"{syn_name}.jpg",
                            syn_img.shape,
                            syn_boxes,
                            folder_name="processed_images"
                        )
                    processed_count += 1

            # Báo tiến độ từng ảnh một
            send_progress_webhook(
                webhook_url,
                job_id,
                idx,
                len(image_files),
                current_file=file_name,
                calipers_detected=total_calipers_detected
            )

        # Xuất Master CSV vào coordinates/calipers_master.csv
        if export_csv:
            csv_file_path = os.path.join(coordinates_dir, "calipers_master.csv")
            with open(csv_file_path, "w", encoding="utf-8") as f_csv:
                f_csv.write("\n".join(master_csv_rows))

        # Xuất Master JSON vào coordinates/calipers_master.json
        if export_json:
            json_file_path = os.path.join(coordinates_dir, "calipers_master.json")
            with open(json_file_path, "w", encoding="utf-8") as f_json:
                json.dump({
                    "job_id": job_id,
                    "total_images": len(image_files),
                    "total_calipers_detected": total_calipers_detected,
                    "items": master_json_items
                }, f_json, indent=2, ensure_ascii=False)

        # Xóa thư mục gốc nếu không yêu cầu
        if not include_original and os.path.exists(orig_images_dir):
            import shutil
            shutil.rmtree(orig_images_dir, ignore_errors=True)

        # Đóng gói zip với đầy đủ 3 thư mục con
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
            "total_images": len(image_files),
            "download_url": f"http://127.0.0.1:8000/download/processed_{job_id}.zip"
        }

        if webhook_url:
            try:
                requests.post(webhook_url, json=result, timeout=10)
                logger.info(f"[JOB {job_id}] Webhook thông báo thành công.")
            except Exception as e:
                logger.error(f"[JOB {job_id}] Webhook thất bại: {e}")

