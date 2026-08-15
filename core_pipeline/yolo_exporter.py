import os

def save_to_yolo_txt(filepath: str, img_shape: tuple, boxes: list[dict]):
    """
    Lưu annotation dưới định dạng YOLO (.txt).
    Dữ liệu format: <class> <x_center> <y_center> <width> <height> (đã chuẩn hóa 0-1)
    """
    class_map = {"plus": 0, "x_mark": 1, "caliper": 2}
    h, w = img_shape[:2]
    
    lines = []
    for b in boxes:
        cls_id = class_map.get(b['name'], 2)  # default to caliper if unknown
        xmin, ymin, xmax, ymax = b['xmin'], b['ymin'], b['xmax'], b['ymax']
        
        # YOLO format requires normalized coordinates
        x_center = ((xmin + xmax) / 2.0) / w
        y_center = ((ymin + ymax) / 2.0) / h
        box_w = (xmax - xmin) / w
        box_h = (ymax - ymin) / h
        
        # Ensure values are within [0, 1] bounds
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        box_w = max(0.0, min(1.0, box_w))
        box_h = max(0.0, min(1.0, box_h))
        
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
        
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
