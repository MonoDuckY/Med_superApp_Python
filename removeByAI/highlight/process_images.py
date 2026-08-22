import cv2
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

def prettify_xml(elem):
    """Giúp định dạng file XML có xuống dòng và thụt lề đẹp mắt"""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")

def highlight_and_extract_all_boxes(img, templates_dir, threshold=0.72):
    """
    Quét đa mẫu, tô đỏ ảnh gốc và gộp chung vào 1 danh sách.
    SỬA LỖI: Chặn tuyệt đối việc trùng lặp tọa độ giữa các class khác nhau (plus, x_mark, caliper).
    """
    all_boxes = []
    if not os.path.exists(templates_dir):
        return img, all_boxes

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    template_files = [f for f in os.listdir(templates_dir) if f.lower().endswith(valid_extensions)]

    if not template_files:
        return img, all_boxes

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    for tpl_name in template_files:
        # 1. Xác định tên class dựa trên tên file template
        if "plus" in tpl_name.lower():
            class_name = "plus"
        elif "x" in tpl_name.lower():
            class_name = "x_mark"
        else:
            class_name = "caliper"

        tpl_path = os.path.join(templates_dir, tpl_name)
        tpl = cv2.imread(tpl_path, cv2.IMREAD_GRAYSCALE)
        h, w = tpl.shape[:2]
        
        _, tpl_mask = cv2.threshold(tpl, 220, 255, cv2.THRESH_BINARY)
        res = cv2.matchTemplate(gray_filtered, tpl, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        
        for pt in zip(*loc[::-1]):
            x_start, y_start = pt[0], pt[1]
            
            # Khử nhiễu vùng biên chứa chữ số ở góc màn hình 1024x768
            if x_start < 80 or y_start > 650:
                continue
                
            if y_start + h > img.shape[0] or x_start + w > img.shape[1]:
                continue

            # Tính toán tọa độ bounding box dự kiến trước để kiểm tra trùng lặp
            xmin = max(0, x_start - 2)
            ymin = max(0, y_start - 2)
            xmax = min(img.shape[1], x_start + w + 2)
            ymax = min(img.shape[0], y_start + h + 2)
            
            # --- ĐOẠN THAY ĐỔI QUAN TRỌNG: KIỂM TRA TRÙNG LẶP TOÀN DIỆN ---
            # Bất kể b['name'] là gì (plus, x_mark hay caliper), nếu tọa độ lệch nhau < 8 pixel
            # thì coi như đã nhận diện xong mục tiêu này ở template trước -> BỎ QUA.
            is_duplicate = False
            for b in all_boxes:
                if abs(b['xmin'] - xmin) < 8 and abs(b['ymin'] - ymin) < 8:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
            # ------------------------------------------------------------

            # Nếu là vị trí mới hoàn toàn, tiến hành phân tích độ sáng cục bộ và tô màu
            gray_roi = gray[y_start:y_start+h, x_start:x_start+w]
            img_roi = img[y_start:y_start+h, x_start:x_start+w]
            
            _, dynamic_mask = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            kernel = np.ones((3, 3), np.uint8)
            tpl_mask_dilated = cv2.dilate(tpl_mask, kernel, iterations=1)
            final_roi_mask = cv2.bitwise_and(tpl_mask_dilated, dynamic_mask)
            
            # Nhuộm đỏ phần dấu trên ảnh gốc
            img_roi[final_roi_mask > 0] = [0, 0, 255]
            
            # Lưu box chuẩn vào danh sách tổng hợp
            all_boxes.append({
                'name': class_name,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })
            
    return img, all_boxes

def save_to_combined_xml(output_path, filename, img_shape, boxes, folder_name="output"):
    """Ghi cấu trúc Pascal VOC XML chứa cả nhãn plus và x_mark chung nhau"""
    annotation = ET.Element('annotation')
    
    ET.SubElement(annotation, 'folder').text = folder_name
    ET.SubElement(annotation, 'filename').text = filename
    ET.SubElement(annotation, 'path').text = os.path.abspath(output_path)
    
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = "Unknown"
    
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(img_shape[1])
    ET.SubElement(size, 'height').text = str(img_shape[0])
    ET.SubElement(size, 'depth').text = str(img_shape[2])
    
    ET.SubElement(annotation, 'segmented').text = "0"
    
    for box in boxes:
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = box['name']
        ET.SubElement(obj, 'pose').text = "Unspecified"
        ET.SubElement(obj, 'truncated').text = "0"
        ET.SubElement(obj, 'difficult').text = "0"
        
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(box['xmin'])
        ET.SubElement(bndbox, 'ymin').text = str(box['ymin'])
        ET.SubElement(bndbox, 'xmax').text = str(box['xmax'])
        ET.SubElement(bndbox, 'ymax').text = str(box['ymax'])
        
    pretty_xml = prettify_xml(annotation)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

def main():
    input_dir = "input"
    templates_dir = "templates"
    
    # Định nghĩa thư mục cha "output"
    base_output_dir = "output"
    
    # Định nghĩa 3 thư mục con nằm BÊN TRONG thư mục "output"
    out_images_dir = os.path.join(base_output_dir, "output_images")
    out_xmls_dir = os.path.join(base_output_dir, "output_xmls")
    out_combined_dir = os.path.join(base_output_dir, "output_combined")
    
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Đã tạo thư mục '{input_dir}'. Hãy bỏ ảnh vào đây và chạy lại.")
        return

    # Tự động tạo cây thư mục một cách có hệ thống
    for d in [base_output_dir, out_images_dir, out_xmls_dir, out_combined_dir, templates_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    if not images:
        print(f"Không tìm thấy ảnh nào trong thư mục '{input_dir}'!")
        return
        
    print(f"Tìm thấy {len(images)} ảnh. Đang tiến hành phân tách và xuất vào thư mục tổng 'output'...")
    
    for filename in images:
        input_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        
        img = cv2.imread(input_path)
        if img is None:
            print(f"Lỗi: Không thể đọc ảnh {filename}")
            continue
            
        img_to_process = img.copy()
        processed_img, all_boxes = highlight_and_extract_all_boxes(img_to_process, templates_dir, threshold=0.6)
        
        # --- GHI DỮ LIỆU VÀO ĐƯỜNG DẪN MỚI ---
        
        # 1. Ghi vào output/output_images/
        img_only_path = os.path.join(out_images_dir, filename)
        cv2.imwrite(img_only_path, processed_img)
        
        # 2. Ghi vào output/output_xmls/
        xml_only_path = os.path.join(out_xmls_dir, f"{base_name}.xml")
        save_to_combined_xml(xml_only_path, filename, img.shape, all_boxes, folder_name="output_xmls")
        
        # 3. Ghi vào output/output_combined/
        combined_img_path = os.path.join(out_combined_dir, filename)
        combined_xml_path = os.path.join(out_combined_dir, f"{base_name}.xml")
        
        cv2.imwrite(combined_img_path, processed_img)
        save_to_combined_xml(combined_xml_path, filename, img.shape, all_boxes, folder_name="output_combined")
        
        print(f"-> Đã xử lý: {filename} ({len(all_boxes)} đối tượng)")
        
    print(f"\n[THÀNH CÔNG] Toàn bộ dữ liệu đã được gom vào thư mục: '{base_output_dir}/'")

if __name__ == "__main__":
    main()
