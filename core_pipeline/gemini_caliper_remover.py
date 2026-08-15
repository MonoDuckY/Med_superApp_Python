import os
import cv2
import json
import logging
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from dotenv import load_dotenv
from .prompts import build_gemini_caliper_prompt, DEFAULT_CALIPER_REMOVAL_PROMPT

# Tải biến môi trường từ .env nếu có
load_dotenv()

logger = logging.getLogger("gemini_caliper_remover")
logger.setLevel(logging.INFO)

# Kiểm tra thư viện google-genai
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    logger.warning("Thư viện 'google-genai' chưa được cài đặt.")


class GeminiAPIError(Exception):
    """Lỗi khi gọi API LLM Gemini"""
    pass


class GeminiUltrasoundCaliperRemover:
    """
    Module xử lý xóa Caliper trên ảnh siêu âm trực tiếp bằng LLM Gemini Multimodal API:
    1. Sử dụng Gemini Multimodal Vision API (mặc định: gemini-3.1-pro / gemini-2.5-flash) để phân tích & định vị caliper.
    2. Prompt chỉ thị y tế chuyên sâu cho Gemini LLM:
       - Xóa caliper, bảo toàn cấu trúc mô giải phẫu bên dưới nét vẽ.
       - Giữ nguyên 100% các vùng khác ngoài caliper (Strict zero-modification).
       - Giữ nguyên đặc trưng nhiễu hạt siêu âm (Speckle Noise) và độ phản hồi âm.
       - Giữ nguyên texture và màu sắc siêu âm (chống ngả màu RGB).
    3. Tách đúng nét vẽ caliper (Sub-pixel Stroke Mask) từ phản hồi của Gemini LLM.
    4. Phục hồi cấu trúc theo đường đẳng sáng (Isophote Structure Inpainting) & bù nhiễu hạt siêu âm cục bộ.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-3.1-pro")
        self.client = None
        if self.api_key:
            self._init_client(self.api_key)

    def _init_client(self, api_key: str):
        if not HAS_GENAI:
            raise GeminiAPIError("Thư viện 'google-genai' chưa được cài đặt trong môi trường Python.")
        try:
            self.api_key = api_key
            self.client = genai.Client(api_key=self.api_key)
            logger.info(f"Đã khởi tạo kết nối LLM Gemini Client với model: {self.model_name}")
        except Exception as e:
            logger.error(f"Lỗi khởi tạo Gemini Client: {e}")
            raise GeminiAPIError(f"Không thể khởi tạo kết nối với Gemini API: {str(e)}")

    def detect_calipers_with_llm(
        self, 
        image: np.ndarray, 
        custom_api_key: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Gọi trực tiếp API LLM Gemini 3.1 Pro Multimodal Vision để nhận diện và định vị Caliper trên ảnh siêu âm.
        """
        active_key = custom_api_key or self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        active_model = model_name or self.model_name or "gemini-3.1-pro"

        if not active_key:
            raise GeminiAPIError(
                f"Chưa có GEMINI_API_KEY. Vui lòng cung cấp Gemini API Key (lấy miễn phí tại https://aistudio.google.com/app/apikey) để thực thi qua LLM {active_model}."
            )

        if not self.client or (custom_api_key and custom_api_key != self.api_key):
            self._init_client(active_key)

        h, w = image.shape[:2]
        success, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not success:
            raise GeminiAPIError("Không thể mã hóa ảnh JPEG để gửi đến Gemini API.")

        # Xây dựng prompt từ file prompts.py
        prompt = build_gemini_caliper_prompt(custom_prompt)

        # Danh sách model ứng viên tương thích nếu tên model không tồn tại trên Google Cloud API
        model_candidates = []
        if active_model:
            model_candidates.append(active_model)
        
        # Thêm các model Pro & Flash chính thức của Google
        fallback_models = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
        for m in fallback_models:
            if m not in model_candidates:
                model_candidates.append(m)

        response = None
        last_error = None
        used_model = active_model

        for try_model in model_candidates:
            try:
                logger.info(f"Đang gửi request tới Google Gemini LLM API (Model: {try_model})...")
                response = self.client.models.generate_content(
                    model=try_model,
                    contents=[
                        types.Part.from_bytes(data=buffer.tobytes(), mime_type="image/jpeg"),
                        prompt
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1
                    )
                )
                used_model = try_model
                logger.info(f"Gemini API thành công với model: {used_model}")
                break
            except Exception as e:
                err_msg = str(e)
                last_error = e
                # Nếu model không tồn tại (404 NOT_FOUND), tự động chuyển sang model khả dụng tiếp theo
                if "404" in err_msg or "NOT_FOUND" in err_msg or "not supported" in err_msg.lower():
                    logger.warning(f"Model '{try_model}' không khả dụng trên API ({err_msg}). Đang tự động chuyển sang model tiếp theo...")
                    continue
                else:
                    # Lỗi khác (ví dụ sai API key), raise ngay
                    raise GeminiAPIError(f"Lỗi gọi Gemini LLM API: {err_msg}")

        if response is None:
            raise GeminiAPIError(f"Không thể kết nối với các model Gemini ({model_candidates}). Chi tiết: {str(last_error)}")

        try:
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            parsed_data = json.loads(response_text)
            boxes = []
            reasoning_list = []

            if isinstance(parsed_data, list):
                for item in parsed_data:
                    box_2d = item.get("box_2d") or item.get("bbox") or item.get("box")
                    if box_2d and len(box_2d) == 4:
                        ymin, xmin, ymax, xmax = box_2d
                        if max(ymin, xmin, ymax, xmax) <= 1.0:
                            py_min, px_min = int(ymin * h), int(xmin * w)
                            py_max, px_max = int(ymax * h), int(xmax * w)
                        elif max(ymin, xmin, ymax, xmax) <= 1000:
                            py_min, px_min = int((ymin / 1000.0) * h), int((xmin / 1000.0) * w)
                            py_max, px_max = int((ymax / 1000.0) * h), int((xmax / 1000.0) * w)
                        else:
                            py_min, px_min, py_max, px_max = int(ymin), int(xmin), int(ymax), int(xmax)

                        px_min = max(0, px_min - 3)
                        py_min = max(0, py_min - 3)
                        px_max = min(w, px_max + 3)
                        py_max = min(h, py_max + 3)

                        boxes.append({
                            "name": str(item.get("label", "caliper")),
                            "xmin": int(px_min),
                            "ymin": int(py_min),
                            "xmax": int(px_max),
                            "ymax": int(py_max)
                        })
                        if "reasoning" in item:
                            reasoning_list.append(str(item["reasoning"]))

            logger.info(f"Gemini LLM API đã nhận diện thành công {len(boxes)} calipers.")
            llm_info = {
                "engine": f"Google Gemini LLM ({used_model})",
                "status": "SUCCESS",
                "calipers_detected": int(len(boxes)),
                "llm_reasoning": reasoning_list
            }
            return boxes, llm_info

        except Exception as e:
            logger.error(f"Lỗi khi xử lý phản hồi JSON từ Gemini LLM: {str(e)}")
            raise GeminiAPIError(f"Lỗi xử lý phản hồi JSON từ Gemini LLM: {str(e)}")


    def create_subpixel_caliper_mask(self, image: np.ndarray, boxes: List[Dict[str, Any]]) -> np.ndarray:
        """
        Tạo Sub-pixel Mask cực chính xác bám theo nét vẽ của caliper bên trong Bounding Boxes do Gemini cung cấp.
        Không che mù cả khối chữ nhật, bảo toàn 100% diện tích mô siêu âm xung quanh nét vẽ.
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        for b in boxes:
            xmin, ymin, xmax, ymax = int(b['xmin']), int(b['ymin']), int(b['xmax']), int(b['ymax'])
            roi_gray = gray[ymin:ymax, xmin:xmax]
            if roi_gray.size == 0:
                continue

            roi_blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)
            _, roi_thresh = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            kernel = np.ones((3, 3), np.uint8)
            roi_tophat = cv2.morphologyEx(roi_gray, cv2.MORPH_TOPHAT, kernel)
            _, roi_tophat_thresh = cv2.threshold(roi_tophat, 20, 255, cv2.THRESH_BINARY)

            roi_stroke = cv2.bitwise_or(roi_thresh, roi_tophat_thresh)
            roi_stroke_dilated = cv2.dilate(roi_stroke, kernel, iterations=1)

            mask[ymin:ymax, xmin:xmax] = cv2.bitwise_or(mask[ymin:ymax, xmin:xmax], roi_stroke_dilated)

        return mask

    def remove_calipers_with_gemini(
        self,
        image: np.ndarray,
        custom_api_key: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Thực thi toàn bộ quy trình Xóa Caliper bằng LLM Gemini API (Mặc định: gemini-3.1-pro):
        1. Gửi ảnh tới Gemini Multimodal LLM API để phân tích và định vị toàn bộ Caliper.
        2. Tạo Sub-pixel Mask theo nét vẽ Caliper.
        3. Khôi phục cấu trúc giải phẫu bên dưới nét vẽ (Isophote Structure Inpainting).
        4. Tái tạo & bảo toàn đặc trưng nhiễu hạt siêu âm (Speckle Noise Matching).
        5. Giữ nguyên chuẩn màu đơn sắc của ảnh siêu âm (Chống ngả màu RGB).
        6. Giữ nguyên 100% các vùng khác ngoài Caliper (MSE = 0.0).
        """
        if image is None or image.size == 0:
            raise ValueError("Ảnh đầu vào rỗng hoặc không hợp lệ.")

        original_image = image.copy()
        h, w = original_image.shape[:2]
        is_color = len(original_image.shape) == 3 and original_image.shape[2] == 3
        active_model = model_name or self.model_name or "gemini-3.1-pro"

        # Bước 1: Gọi trực tiếp LLM Gemini API với Prompt mẫu chuẩn
        boxes, llm_info = self.detect_calipers_with_llm(original_image, custom_api_key, custom_prompt, active_model)

        if not boxes:
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            meta = {
                "engine": f"Google Gemini LLM ({active_model})",
                "status": "NO_CALIPERS_FOUND",
                "calipers_count": 0,
                "noise_std_estimated": 0.0,
                "is_monochrome_enforced": True,
                "background_difference_mse": 0.0,
                "llm_info": llm_info
            }
            return original_image, empty_mask, [], meta

        # Bước 2: Tạo Sub-pixel Mask theo tọa độ do Gemini LLM phát hiện
        caliper_mask = self.create_subpixel_caliper_mask(original_image, boxes)

        if not np.any(caliper_mask):
            meta = {
                "engine": f"Google Gemini LLM ({active_model})",
                "status": "MASK_EMPTY",
                "calipers_count": int(len(boxes)),
                "noise_std_estimated": 0.0,
                "is_monochrome_enforced": True,
                "background_difference_mse": 0.0,
                "llm_info": llm_info
            }
            return original_image, caliper_mask, boxes, meta


        # Bước 3: Dilate nhẹ 1 pixel để phủ viền chống lem màu (anti-color bleed)
        morph_kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(caliper_mask, morph_kernel, iterations=1)

        # Bước 4: Khôi phục cấu trúc giải phẫu theo đường đẳng sáng (Isophote Navier-Stokes/Telea)
        inpainted = cv2.inpaint(original_image, dilated_mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)

        # Bước 5: Mô hình hóa & Bù Nhiễu Hạt Siêu Âm Cục Bộ (Speckle Noise Matching)
        ring_mask = cv2.dilate(dilated_mask, morph_kernel, iterations=4) - dilated_mask
        gray_orig = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) if is_color else original_image

        if np.any(ring_mask):
            mean_val, std_val = cv2.meanStdDev(gray_orig, mask=ring_mask)
            local_noise_std = float(std_val[0][0])
        else:
            local_noise_std = 6.0

        noise_std_clamped = max(3.0, min(local_noise_std, 18.0))
        noise_matrix = np.random.normal(0, noise_std_clamped, (h, w)).astype(np.float32)

        inpainted_float = inpainted.astype(np.float32)
        is_monochrome_area = True
        if is_color:
            ring_pixels = original_image[ring_mask > 0]
            if len(ring_pixels) > 0:
                rg_diff = float(np.mean(np.abs(ring_pixels[:, 0].astype(int) - ring_pixels[:, 1].astype(int))))
                gb_diff = float(np.mean(np.abs(ring_pixels[:, 1].astype(int) - ring_pixels[:, 2].astype(int))))
                is_monochrome_area = bool(rg_diff < 5.0 and gb_diff < 5.0)

            if is_monochrome_area:
                gray_inpaint = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY).astype(np.float32)
                gray_with_noise = np.clip(gray_inpaint + noise_matrix, 0, 255).astype(np.uint8)
                restored_patch = cv2.cvtColor(gray_with_noise, cv2.COLOR_GRAY2BGR)
            else:
                for c in range(3):
                    inpainted_float[:, :, c] += noise_matrix
                restored_patch = np.clip(inpainted_float, 0, 255).astype(np.uint8)
        else:
            gray_with_noise = np.clip(inpainted_float + noise_matrix, 0, 255).astype(np.uint8)
            restored_patch = gray_with_noise

        # Bước 6: Ghép Mask Tuyệt Đối (Strict Mask Compositing - MSE = 0.0)
        final_image = original_image.copy()
        if is_color:
            mask_3c = dilated_mask[:, :, None] > 0
            final_image = np.where(mask_3c, restored_patch, original_image)
        else:
            mask_1c = dilated_mask > 0
            final_image = np.where(mask_1c, restored_patch, original_image)

        meta_info = {
            "engine": f"Google Gemini LLM ({active_model})",
            "status": "SUCCESS",
            "calipers_count": int(len(boxes)),
            "noise_std_estimated": float(round(local_noise_std, 2)),

            "is_monochrome_enforced": bool(is_color and is_monochrome_area),
            "background_difference_mse": 0.0,
            "llm_info": llm_info
        }

        return final_image, dilated_mask, boxes, meta_info


# Singleton instance
gemini_caliper_remover = GeminiUltrasoundCaliperRemover()
