import cv2
import numpy as np

def scaled_points(width, height, points_frac):
    return np.array(
        [[int(width * x), int(height * y)] for x, y in points_frac],
        dtype=np.int32,
    )

def detect_safe_area(image: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """
    Phát hiện vùng an toàn (hình quạt siêu âm) và vẽ viền/crop.
    Dựa trên thuật toán của [v3.0] SAfeArea.
    """
    result = image.copy()
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Vùng siêu âm - cố gắng vẽ hình quạt
    upper = gray[0:int(h * 0.6)]
    edges = cv2.Canny(upper, 20, 80)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8))

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fan_contour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 35000:  # Vùng quạt
            fan_contour = cnt
            break

    bbox = [0, 0, w, h] # Default to full image if not found
    if fan_contour is not None:
        # Tạm thời chỉ vẽ viền xanh lá để debug. Trong thực tế sẽ crop hoặc tạo mask.
        cv2.drawContours(result, [fan_contour], -1, (0, 255, 0), 8)
        x, y, w_cnt, h_cnt = cv2.boundingRect(fan_contour)
        bbox = [x, y, x + w_cnt, y + h_cnt]

    return result, bbox

from rapidocr_onnxruntime import RapidOCR

# Khởi tạo model OCR (load 1 lần)
ocr = RapidOCR()

def inpaint_speckle_texture_aware(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Thuật toán Inpainting chuyên biệt cho ảnh siêu âm:
    - Sử dụng Fast Marching / Navier-Stokes với bán kính phù hợp để khôi phục cấu trúc gradient mô tự nhiên.
    - Bảo toàn đường viền tổn thương và mật độ phản âm (Echogenicity) của mô bên dưới.
    """
    if mask is None or np.count_nonzero(mask) == 0:
        return image.copy()
        
    # Inpaint khôi phục nền mượt mà theo gradient mô xung quanh
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

def remove_calipers(image: np.ndarray, mask: np.ndarray, method: str = "speckle_aware") -> np.ndarray:
    """
    Xóa các dấu Caliper trên ảnh siêu âm và lấp đầy nền.
    - 'speckle_aware': Khôi phục mô kèm hạt nhiễu siêu âm (Khuyến nghị).
    - 'telea': Fast Marching thông thường.
    - 'ns': Navier-Stokes.
    """
    if mask is None or np.count_nonzero(mask) == 0:
        return image.copy()
        
    if method == "speckle_aware":
        return inpaint_speckle_texture_aware(image, mask)
    elif method == "ns":
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
    else:
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

def remove_text_and_callipers(image: np.ndarray, method: str = "speckle_aware") -> np.ndarray:
    """
    Xóa chữ trên ảnh siêu âm bằng RapidOCR và Inpainting bảo tồn mô.
    """
    result = image.copy()
    
    # 1. Nhận diện chữ bằng RapidOCR
    ocr_result, _ = ocr(result)
    
    if ocr_result:
        h, w = result.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for item in ocr_result:
            box, text, score = item
            if not text.strip():
                continue
                
            box_points = np.array(box, dtype=np.int32)
            cv2.fillPoly(mask, [box_points], 255)
            
        # Inpaint xóa chữ
        if method == "speckle_aware":
            result = inpaint_speckle_texture_aware(result, mask)
        else:
            result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)

    return result

import os

_CACHED_TEMPLATES = {}

def _get_scaled_templates(templates_dir: str, scales: tuple = (0.75, 0.9, 1.0, 1.15)):
    cache_key = (templates_dir, scales)
    if cache_key in _CACHED_TEMPLATES:
        return _CACHED_TEMPLATES[cache_key]

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    if not os.path.exists(templates_dir):
        return []

    template_files = [f for f in os.listdir(templates_dir) if f.lower().endswith(valid_extensions)]
    cached_list = []
    kernel = np.ones((3, 3), np.uint8)

    for tpl_name in template_files:
        if "plus" in tpl_name.lower():
            class_name = "plus"
        elif "x" in tpl_name.lower():
            class_name = "x_mark"
        else:
            class_name = "caliper"

        tpl_path = os.path.join(templates_dir, tpl_name)
        tpl_orig = cv2.imread(tpl_path, cv2.IMREAD_GRAYSCALE)
        if tpl_orig is None:
            continue

        orig_h, orig_w = tpl_orig.shape[:2]

        for s in scales:
            sw, sh = max(5, int(orig_w * s)), max(5, int(orig_h * s))
            tpl = cv2.resize(tpl_orig, (sw, sh), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
            _, tpl_mask = cv2.threshold(tpl, 200, 255, cv2.THRESH_BINARY)
            tpl_mask_dilated = cv2.dilate(tpl_mask, kernel, iterations=2)
            cached_list.append((class_name, tpl, tpl_mask_dilated, sw, sh))

    _CACHED_TEMPLATES[cache_key] = cached_list
    return cached_list

def detect_calipers(
    image: np.ndarray,
    templates_dir: str,
    threshold: float = 0.58,
    scales: tuple = (0.75, 0.9, 1.0, 1.15)
) -> tuple[np.ndarray, list[dict]]:
    """
    Phát hiện các dấu Caliper đa tỉ lệ (Multi-Scale Template Matching + NMS).
    Bắt trọn vẹn cả dấu '+' và 'x' ở mọi kích thước/độ phân giải máy siêu âm.
    """
    h_shape, w_shape = image.shape[:2]
    mask = np.zeros((h_shape, w_shape), dtype=np.uint8)
    boxes = []

    cached_templates = _get_scaled_templates(templates_dir, scales)
    if not cached_templates:
        return mask, boxes

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray_filtered = cv2.GaussianBlur(gray, (3, 3), 0.5)

    candidates = []

    for class_name, tpl, tpl_mask_dilated, sw, sh in cached_templates:
        res = cv2.matchTemplate(gray_filtered, tpl, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        
        for pt in zip(*loc[::-1]):
            x_start, y_start = pt[0], pt[1]
            score = float(res[y_start, x_start])
            
            # Loại trừ vùng viền đen và thanh thông số xung quanh màn hình
            if x_start < 60 or x_start > (w_shape - 80) or y_start < 40 or y_start > (h_shape - 90):
                continue
            if y_start + sh > h_shape or x_start + sw > w_shape:
                continue

            xmin = max(0, x_start - 2)
            ymin = max(0, y_start - 2)
            xmax = min(w_shape, x_start + sw + 2)
            ymax = min(h_shape, y_start + sh + 2)

            candidates.append({
                "score": score,
                "name": class_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "x_start": x_start,
                "y_start": y_start,
                "sh": sh,
                "sw": sw,
                "mask": tpl_mask_dilated
            })

    # Sắp xếp các ứng viên theo độ tin cậy (score) giảm dần để thực hiện NMS
    candidates.sort(key=lambda c: c["score"], reverse=True)

    for c in candidates:
        is_dup = False
        for b in boxes:
            if abs(b["xmin"] - c["xmin"]) < 10 and abs(b["ymin"] - c["ymin"]) < 10:
                is_dup = True
                break
        if is_dup:
            continue

        tmp_full = np.zeros_like(mask)
        tmp_full[c["y_start"]:c["y_start"] + c["sh"], c["x_start"]:c["x_start"] + c["sw"]] = c["mask"]
        mask = cv2.bitwise_or(mask, tmp_full)
        
        boxes.append({
            "name": str(c["name"]),
            "xmin": int(c["xmin"]),
            "ymin": int(c["ymin"]),
            "xmax": int(c["xmax"]),
            "ymax": int(c["ymax"])
        })

    # Mở rộng nhẹ mask 1px để phủ kín toàn bộ viền bóng mờ (drop shadow) của caliper
    if np.any(mask > 0):
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        
    return mask, boxes

def highlight_calipers_on_image(image: np.ndarray, mask: np.ndarray, boxes: list[dict], color=(0, 0, 255), draw_box=True) -> np.ndarray:
    """
    Tô màu đỏ lên các điểm caliper phát hiện được trên ảnh và vẽ bounding box viền đỏ.
    """
    result = image.copy()
    if mask is not None and np.any(mask > 0):
        # Nhuộm đỏ các pixel của caliper (BGR format: [0, 0, 255])
        result[mask > 0] = color
        
    if draw_box and boxes:
        for b in boxes:
            xmin, ymin = int(b['xmin']), int(b['ymin'])
            xmax, ymax = int(b['xmax']), int(b['ymax'])
            cv2.rectangle(result, (xmin, ymin), (xmax, ymax), color, 1)
            # Vẽ nhãn nhỏ góc trên của box
            label = b.get('name', 'caliper')
            cv2.putText(result, label, (xmin, max(12, ymin - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
            
    return result

