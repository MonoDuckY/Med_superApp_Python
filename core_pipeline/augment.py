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

def draw_synthetic_caliper_pair(image: np.ndarray, safe_bbox: list[int] = None) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Sinh một cặp dấu Caliper giả lập chuẩn máy siêu âm (GE, Philips, Siemens, Mindray).
    - Cặp dấu: 2 điểm chữ thập '+' hoặc 'x'
    - Đường đo nét đứt (Dashed line) nối 2 điểm
    - Nhãn đo lường giả lập (ví dụ: 'D 14.8mm')
    Trả về: (augmented_image, caliper_mask, new_boxes)
    """
    aug = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Giới hạn vùng sinh caliper trong vùng an toàn (Safe Area) hoặc giữa ảnh
    if safe_bbox and len(safe_bbox) == 4:
        bx1, by1, bx2, by2 = safe_bbox
        min_x = max(int(w * 0.15), int(bx1 + (bx2 - bx1) * 0.2))
        max_x = min(int(w * 0.85), int(bx1 + (bx2 - bx1) * 0.8))
        min_y = max(int(h * 0.15), int(by1 + (by2 - by1) * 0.2))
        max_y = min(int(h * 0.85), int(by1 + (by2 - by1) * 0.8))
    else:
        min_x, max_x = int(w * 0.3), int(w * 0.7)
        min_y, max_y = int(h * 0.3), int(h * 0.7)
        
    if min_x >= max_x: min_x, max_x = int(w * 0.25), int(w * 0.75)
    if min_y >= max_y: min_y, max_y = int(h * 0.25), int(h * 0.75)
    
    cx = np.random.randint(min_x, max_x)
    cy = np.random.randint(min_y, max_y)
    dist = np.random.randint(40, 110)
    angle = np.random.uniform(0, 2 * np.pi)
    
    p1 = (int(cx - (dist / 2) * np.cos(angle)), int(cy - (dist / 2) * np.sin(angle)))
    p2 = (int(cx + (dist / 2) * np.cos(angle)), int(cy + (dist / 2) * np.sin(angle)))
    
    # Kiểm tra biên màn hình
    p1 = (max(20, min(w - 20, p1[0])), max(20, min(h - 20, p1[1])))
    p2 = (max(20, min(w - 20, p2[0])), max(20, min(h - 20, p2[1])))
    
    color = (255, 255, 255) if np.random.random() > 0.3 else (210, 255, 255)
    mark_type = np.random.choice(['plus', 'x_mark'])
    sz = 6
    new_boxes = []
    
    def draw_pt(pt, m_type):
        x, y = pt
        if m_type == 'plus':
            cv2.line(aug, (x - sz, y), (x + sz, y), color, 1)
            cv2.line(aug, (x, y - sz), (x, y + sz), color, 1)
            cv2.line(mask, (x - sz, y), (x + sz, y), 255, 1)
            cv2.line(mask, (x, y - sz), (x, y + sz), 255, 1)
        else:
            cv2.line(aug, (x - sz, y - sz), (x + sz, y + sz), color, 1)
            cv2.line(aug, (x - sz, y + sz), (x + sz, y - sz), color, 1)
            cv2.line(mask, (x - sz, y - sz), (x + sz, y + sz), 255, 1)
            cv2.line(mask, (x - sz, y + sz), (x + sz, y - sz), 255, 1)
        new_boxes.append({
            'name': m_type,
            'xmin': max(0, x - sz - 2),
            'ymin': max(0, y - sz - 2),
            'xmax': min(w, x + sz + 2),
            'ymax': min(h, y + sz + 2)
        })

    # Vẽ đường nét đứt nối 2 điểm
    num_steps = 14
    for s in range(num_steps):
        if s % 2 == 0:
            s_pt = (int(p1[0] + (p2[0] - p1[0]) * (s / num_steps)), int(p1[1] + (p2[1] - p1[1]) * (s / num_steps)))
            e_pt = (int(p1[0] + (p2[0] - p1[0]) * ((s + 1) / num_steps)), int(p1[1] + (p2[1] - p1[1]) * ((s + 1) / num_steps)))
            cv2.line(aug, s_pt, e_pt, color, 1)
            cv2.line(mask, s_pt, e_pt, 255, 1)
            
    draw_pt(p1, mark_type)
    draw_pt(p2, mark_type)
    
    # Nhãn khoảng cách
    dist_mm = f"D {dist * 0.185:.1f}mm"
    cv2.putText(aug, dist_mm, (p1[0] + 8, p1[1] - 4), cv2.FONT_HERSHEY_PLAIN, 0.85, color, 1, cv2.LINE_AA)
    
    return aug, mask, new_boxes

def augment_with_synthetic_calipers(image: np.ndarray, existing_boxes: list[dict] = None) -> list[tuple[np.ndarray, list[dict], str]]:
    """
    Sinh các biến thể có thêm Caliper giả lập để làm giàu dữ liệu và chống Shortcut Learning.
    """
    if existing_boxes is None:
        existing_boxes = []
        
    aug_img, _, syn_boxes = draw_synthetic_caliper_pair(image)
    combined_boxes = list(existing_boxes) + syn_boxes
    
    return [(aug_img, combined_boxes, "aug_synthetic_caliper")]

