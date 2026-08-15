import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

def prettify_xml(elem):
    """Giúp định dạng file XML có xuống dòng và thụt lề đẹp mắt"""
    rough_string = ET.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")

def build_xml_element(filename, img_shape, boxes, folder_name="output", file_path=""):
    """Tạo đối tượng ElementTree XML chuẩn Pascal VOC cho các caliper bounding boxes."""
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = folder_name
    ET.SubElement(annotation, "filename").text = filename
    ET.SubElement(annotation, "path").text = file_path if file_path else filename

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Ultrasound_Caliper_Dataset"

    size = ET.SubElement(annotation, "size")
    if len(img_shape) >= 2:
        ET.SubElement(size, "width").text = str(img_shape[1])
        ET.SubElement(size, "height").text = str(img_shape[0])
    if len(img_shape) >= 3:
        ET.SubElement(size, "depth").text = str(img_shape[2])
    else:
        ET.SubElement(size, "depth").text = "1"

    ET.SubElement(annotation, "segmented").text = "0"

    for box in boxes:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = box.get("name", "caliper")
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(box["xmin"]))
        ET.SubElement(bndbox, "ymin").text = str(int(box["ymin"]))
        ET.SubElement(bndbox, "xmax").text = str(int(box["xmax"]))
        ET.SubElement(bndbox, "ymax").text = str(int(box["ymax"]))

    return annotation

def generate_xml_string(filename, img_shape, boxes, folder_name="output"):
    """Sinh chuỗi Pascal VOC XML dạng string để gửi về UI"""
    elem = build_xml_element(filename, img_shape, boxes, folder_name)
    return prettify_xml(elem)

def save_to_combined_xml(output_path, filename, img_shape, boxes, folder_name="output"):
    """Lưu file Pascal VOC XML xuống đĩa"""
    elem = build_xml_element(filename, img_shape, boxes, folder_name, os.path.abspath(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(prettify_xml(elem))
