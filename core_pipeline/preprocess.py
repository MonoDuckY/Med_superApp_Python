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
