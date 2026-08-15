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

def remove_text_and_callipers(image: np.ndarray) -> np.ndarray:
    """
    Xóa chữ và các thước đo (Calliper) trên ảnh.
    Sử dụng RapidOCR để nhận diện text và inpaint để xóa.
    """
    result = image.copy()
    
    # 1. Nhận diện chữ bằng RapidOCR
    ocr_result, _ = ocr(result)
    
    if ocr_result:
        for item in ocr_result:
            box, text, score = item
            if not text.strip():
                continue
                
            # box có format [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            box_points = np.array(box, dtype=np.int32)
            
            # Tạo mask cho phần chữ để inpaint
            h, w = result.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [box_points], 255)
            
            # Dùng thuật toán inpaint để xóa chữ và "lấp đầy" nền
            result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)

    return result

import os

def detect_calipers(image: np.ndarray, templates_dir: str, threshold: float = 0.62) -> tuple[np.ndarray, list[dict]]:
    """
    Phát hiện các dấu Caliper bằng Template Matching.
    Trả về mask chứa vị trí các caliper và list các bounding boxes.
    """
    h_shape, w_shape = image.shape[:2]
    mask = np.zeros((h_shape, w_shape), dtype=np.uint8)
    boxes = []
    
    if not os.path.exists(templates_dir):
        return mask, boxes

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    template_files = [f for f in os.listdir(templates_dir) if f.lower().endswith(valid_extensions)]

    if not template_files:
        return mask, boxes

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    for tpl_name in template_files:
        if "plus" in tpl_name.lower():
            class_name = "plus"
        elif "x" in tpl_name.lower():
            class_name = "x_mark"
        else:
            class_name = "caliper"

        tpl_path = os.path.join(templates_dir, tpl_name)
        tpl = cv2.imread(tpl_path, cv2.IMREAD_GRAYSCALE)
        if tpl is None: continue
        h, w = tpl.shape[:2]
        
        _, tpl_mask = cv2.threshold(tpl, 220, 255, cv2.THRESH_BINARY)
        res = cv2.matchTemplate(gray_filtered, tpl, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        
        for pt in zip(*loc[::-1]):
            x_start, y_start = pt[0], pt[1]
            if x_start < 80 or y_start > (h_shape - 118): continue
            if y_start + h > h_shape or x_start + w > w_shape: continue

            xmin = max(0, x_start - 2)
            ymin = max(0, y_start - 2)
            xmax = min(w_shape, x_start + w + 2)
            ymax = min(h_shape, y_start + h + 2)
            
            is_duplicate = False
            for b in boxes:
                if abs(b['xmin'] - xmin) < 8 and abs(b['ymin'] - ymin) < 8:
                    is_duplicate = True
                    break
            if is_duplicate: continue

            gray_roi = gray[y_start:y_start+h, x_start:x_start+w]
            _, dynamic_mask = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            kernel = np.ones((3, 3), np.uint8)
            tpl_mask_dilated = cv2.dilate(tpl_mask, kernel, iterations=2)
            final_roi_mask = cv2.bitwise_and(tpl_mask_dilated, dynamic_mask)
            final_roi_mask = cv2.dilate(final_roi_mask, kernel, iterations=1)

            tmp_full_mask = np.zeros_like(mask)
            tmp_full_mask[y_start:y_start+h, x_start:x_start+w] = final_roi_mask
            
            mask = cv2.bitwise_or(mask, tmp_full_mask)
            boxes.append({'name': class_name, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
            
    return mask, boxes

def remove_calipers_preserve_texture(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Xóa caliper dựa trên mask, nhưng giữ nguyên cấu trúc phía bên trong caliper,
    giữ lại toàn bộ cấu trúc của các vùng khác, giữ nguyên đặc trưng nhiễu hạt (speckle noise).
    Giữ nguyên các texture bên trong các caliper và không được tự động ngả màu rgb.
    """
    if not np.any(mask):
        return image
        
    # Bước 1: Dilate mask để đảm bảo bao phủ hoàn toàn viền màu của caliper,
    # tránh tình trạng color bleeding (lem màu rgb của caliper vào trong vùng inpaint).
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Bước 2: Inpaint để nội suy cấu trúc (structure) làm nền phía bên trong caliper.
    # Telea algorithm giúp giữ cấu trúc khá tốt.
    inpainted = cv2.inpaint(image, dilated_mask, 3, cv2.INPAINT_TELEA)
    
    # Bước 3: Lấy độ lệch chuẩn (mức độ nhiễu) từ vùng cấu trúc xung quanh caliper.
    ring_mask = cv2.dilate(dilated_mask, kernel, iterations=5) - dilated_mask
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if np.any(ring_mask):
        mean_val, std_val = cv2.meanStdDev(gray_image, mask=ring_mask)
        noise_std = std_val[0][0]
    else:
        noise_std = 5.0
        
    # Bước 4: Tạo nhiễu đơn sắc (monochromatic noise) để giữ nguyên texture
    # Nhiễu đơn sắc khi cộng vào 3 kênh sẽ không làm thay đổi hay ngả màu RGB của ảnh gốc.
    noise = np.random.normal(0, noise_std, gray_image.shape)
    
    # Bước 5: Thêm nhiễu vào vùng inpaint (cộng nhiễu cho cả 3 kênh như nhau)
    result = inpainted.astype(np.float32)
    for c in range(3):
        result[:, :, c] += noise
        
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Bước 6: Ghép vùng đã xóa caliper vào ảnh gốc
    # np.where giúp đảm bảo giữ lại toàn bộ cấu trúc của các vùng khác, không chỉnh sửa bất cứ điều gì
    final_image = np.where(dilated_mask[:, :, None] > 0, result, image)
    
    return final_image

from .gemini_caliper_remover import gemini_caliper_remover, GeminiUltrasoundCaliperRemover

def remove_calipers_with_gemini(
    image: np.ndarray, 
    api_key: str = None,
    prompt: str = None
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    """
    Hàm tiện ích xóa Caliper bằng Gemini LLM:
    - Sử dụng prompt mẫu chuẩn hoặc prompt tùy chỉnh
    - Giữ nguyên 100% cấu trúc, texture, hạt siêu âm bên trong & ngoài caliper
    - Chống ngả màu RGB
    """
    return gemini_caliper_remover.remove_calipers_with_gemini(image, custom_api_key=api_key, custom_prompt=prompt)


