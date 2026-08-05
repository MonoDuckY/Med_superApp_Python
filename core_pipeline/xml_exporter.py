import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

def prettify_xml(elem):
    rough_string = ET.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")

def save_to_combined_xml(output_path, filename, img_shape, boxes, folder_name="output"):
    """Save Pascal VOC XML for detected caliper boxes."""
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = folder_name
    ET.SubElement(annotation, "filename").text = filename
    ET.SubElement(annotation, "path").text = os.path.abspath(output_path)

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    # shape is (height, width, channels)
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
        ET.SubElement(bndbox, "xmin").text = str(box["xmin"])
        ET.SubElement(bndbox, "ymin").text = str(box["ymin"])
        ET.SubElement(bndbox, "xmax").text = str(box["xmax"])
        ET.SubElement(bndbox, "ymax").text = str(box["ymax"])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(prettify_xml(annotation))
