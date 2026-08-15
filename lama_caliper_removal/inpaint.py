import cv2
import numpy as np
import os
import torch
# Force CPU to avoid Windows CUDA backend errors
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.cuda.is_available = lambda: False

from simple_lama_inpainting import SimpleLama
from PIL import Image

# Initialize LaMa model (loads weights on first run)
lama = SimpleLama()

def detect_calipers(image: np.ndarray, templates_dir: str, threshold: float = 0.62) -> tuple[np.ndarray, list[dict]]:
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
            # Ignore boundaries to reduce false positives
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

def inpaint_calipers(image: np.ndarray, templates_dir: str) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Detects calipers and removes them using LaMa inpainting model.
    Uses Downscale-Inpaint-Upscale blending to massively speed up CPU inference.
    """
    # 1. Detect calipers and get exact mask on ORIGINAL high-res image
    mask, boxes = detect_calipers(image, templates_dir)
    
    if not np.any(mask):
        return image, mask, boxes

    # Dilate mask slightly for better inpainting results at edges
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    
    # 2. Downscale image and mask for LaMa (max dimension 768px)
    h_orig, w_orig = image.shape[:2]
    max_dim = 768
    
    scale_factor = 1.0
    if max(h_orig, w_orig) > max_dim:
        scale_factor = max_dim / float(max(h_orig, w_orig))
        
    new_w = int(w_orig * scale_factor)
    new_h = int(h_orig * scale_factor)
    
    small_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(dilated_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # 3. Convert to PIL Image for SimpleLama
    image_pil = Image.fromarray(cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(small_mask).convert('L')
    
    # 4. Perform Fast Inpainting using LaMa on small image
    result_pil = lama(image_pil, mask_pil)
    small_result_cv2 = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
    
    # 5. Upscale the inpainted image back to original resolution
    large_inpainted_cv2 = cv2.resize(small_result_cv2, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    
    # 6. Blend ONLY the inpainted regions back to the high-res original image
    # (keeps original sharpness everywhere else)
    mask_3d = cv2.cvtColor(dilated_mask, cv2.COLOR_GRAY2BGR) / 255.0
    final_result = (image * (1.0 - mask_3d) + large_inpainted_cv2 * mask_3d).astype(np.uint8)
    
    return final_result, dilated_mask, boxes
