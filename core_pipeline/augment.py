import cv2
import numpy as np

def _flip_boxes_horizontal(boxes, img_w):
    new_boxes = []
    for b in boxes:
        new_b = b.copy()
        new_b['xmin'] = img_w - b['xmax']
        new_b['xmax'] = img_w - b['xmin']
        new_boxes.append(new_b)
    return new_boxes

def _flip_boxes_vertical(boxes, img_h):
    new_boxes = []
    for b in boxes:
        new_b = b.copy()
        new_b['ymin'] = img_h - b['ymax']
        new_b['ymax'] = img_h - b['ymin']
        new_boxes.append(new_b)
    return new_boxes

def augment_image_and_xml(image: np.ndarray, boxes: list[dict]) -> list[tuple[np.ndarray, list[dict], str]]:
    """
    Tạo ra 3 phiên bản augmented của ảnh và tính toán lại bounding boxes tương ứng.
    Trả về list các tuple: (new_image, new_boxes, suffix_name)
    - Flip Horizontal (Lật ngang)
    - Flip Vertical (Lật dọc / Xoay xuống)
    - Rotate 180 (Lật cả ngang và dọc)
    """
    h, w = image.shape[:2]
    augmented_results = []
    
    # 1. Flip Horizontal (ngang trái/phải)
    img_flip_h = cv2.flip(image, 1)
    boxes_flip_h = _flip_boxes_horizontal(boxes, w)
    augmented_results.append((img_flip_h, boxes_flip_h, "aug_flip_h"))
    
    # 2. Flip Vertical (xoay xuống)
    img_flip_v = cv2.flip(image, 0)
    boxes_flip_v = _flip_boxes_vertical(boxes, h)
    augmented_results.append((img_flip_v, boxes_flip_v, "aug_flip_v"))
    
    # 3. Rotate 180 (Flip both)
    img_rot_180 = cv2.flip(image, -1)
    boxes_rot_180 = _flip_boxes_vertical(_flip_boxes_horizontal(boxes, w), h)
    augmented_results.append((img_rot_180, boxes_rot_180, "aug_rot_180"))
    
    return augmented_results
