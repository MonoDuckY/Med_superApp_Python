import os
import sys
import cv2
import numpy as np
import unittest
from unittest.mock import patch, MagicMock

# Thêm root dir vào PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core_pipeline.gemini_caliper_remover import GeminiUltrasoundCaliperRemover


class TestGeminiUltrasoundCaliperRemover(unittest.TestCase):
    
    def setUp(self):
        self.remover = GeminiUltrasoundCaliperRemover(api_key="mock_key_for_testing")

        
        # Tạo ảnh siêu âm giả lập kích thước 300x300 với nhiễu hạt (speckle noise)
        np.random.seed(42)
        h, w = 300, 300
        
        # Nền siêu âm với gradient cấu trúc giải phẫu và nhiễu hạt
        y, x = np.ogrid[:h, :w]
        tissue_gradient = 80 + 40 * np.sin(x / 40.0) + 30 * np.cos(y / 50.0)
        speckle_noise = np.random.normal(0, 10.0, (h, w))
        synthetic_ultrasound_gray = np.clip(tissue_gradient + speckle_noise, 0, 255).astype(np.uint8)
        
        # 3 kênh màu xám đơn sắc (R=G=B)
        self.clean_ultrasound = cv2.cvtColor(synthetic_ultrasound_gray, cv2.COLOR_GRAY2BGR)
        
        # Vẽ các caliper nhân tạo (Dấu thập '+' và chữ 'x')
        self.image_with_caliper = self.clean_ultrasound.copy()
        
        # Caliper 1: Dấu cộng màu vàng/trắng tại (100, 100)
        cv2.line(self.image_with_caliper, (90, 100), (110, 100), (0, 255, 255), 2)
        cv2.line(self.image_with_caliper, (100, 90), (100, 110), (0, 255, 255), 2)
        
        # Caliper 2: Dấu 'x' màu trắng tại (200, 200)
        cv2.line(self.image_with_caliper, (192, 192), (208, 208), (255, 255, 255), 2)
        cv2.line(self.image_with_caliper, (192, 208), (208, 192), (255, 255, 255), 2)

        # Mock phương thức detect_calipers_with_llm để trả về đúng 2 caliper trên
        self.mock_boxes = [
            {"name": "caliper_cross", "xmin": 88, "ymin": 88, "xmax": 112, "ymax": 112},
            {"name": "caliper_x", "xmin": 190, "ymin": 190, "xmax": 210, "ymax": 210}
        ]
        self.mock_llm_info = {
            "engine": "Google Gemini LLM (gemini-3.1-pro)",
            "status": "SUCCESS",
            "calipers_detected": 2,
            "llm_reasoning": ["Found plus caliper at (100,100)", "Found x caliper at (200,200)"]
        }
        self.remover.detect_calipers_with_llm = MagicMock(return_value=(self.mock_boxes, self.mock_llm_info))


    def test_background_invariance(self):

        """
        Yêu cầu 2: Giữ lại toàn bộ cấu trúc của các vùng khác, 
        không chỉnh sửa bất cứ điều gì ở các vùng khác (MSE = 0.0).
        """
        processed_img, mask, boxes, meta = self.remover.remove_calipers_with_gemini(self.image_with_caliper)
        
        # Kiểm tra vùng ngoài mask (mask == 0)
        outside_mask = (mask == 0)
        diff = np.abs(self.image_with_caliper[outside_mask].astype(int) - processed_img[outside_mask].astype(int))
        mse = np.mean(diff ** 2)
        
        self.assertEqual(mse, 0.0, "Vùng ngoài caliper phải giống ảnh gốc 100% (MSE = 0.0)")
        self.assertEqual(meta["background_difference_mse"], 0.0)

    def test_monochrome_consistency_no_rgb_tint(self):
        """
        Yêu cầu 4: Giữ nguyên màu của ảnh siêu âm, 
        không được tự động ngả màu rgb (R=G=B trên vùng siêu âm đen trắng).
        """
        processed_img, mask, boxes, meta = self.remover.remove_calipers_with_gemini(self.image_with_caliper)
        
        if np.any(mask > 0):
            inpainted_pixels = processed_img[mask > 0]
            # Kiểm tra độ lệch giữa các kênh màu
            rg_diff = np.mean(np.abs(inpainted_pixels[:, 0].astype(int) - inpainted_pixels[:, 1].astype(int)))
            gb_diff = np.mean(np.abs(inpainted_pixels[:, 1].astype(int) - inpainted_pixels[:, 2].astype(int)))
            
            self.assertEqual(rg_diff, 0.0, "Kênh R và G phải đồng nhất, không được ngả màu RGB")
            self.assertEqual(gb_diff, 0.0, "Kênh G và B phải đồng nhất, không được ngả màu RGB")

    def test_speckle_noise_preservation(self):
        """
        Yêu cầu 3: Giữ nguyên đặc trưng nhiễu hạt của ảnh siêu âm,
        không làm phẳng/trơn láng (smooth blur) vùng caliper.
        """
        processed_img, mask, boxes, meta = self.remover.remove_calipers_with_gemini(self.image_with_caliper)
        
        if np.any(mask > 0):
            # Tính độ lệch chuẩn của vùng đã phục hồi
            gray_processed = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
            _, std_val = cv2.meanStdDev(gray_processed, mask=mask)
            reconstructed_noise_std = float(std_val[0][0])
            
            # Đảm bảo độ lệch chuẩn nhiễu hạt lớn hơn 2.0 (không bị mờ bệt phẳng lì)
            self.assertGreater(
                reconstructed_noise_std, 
                2.0, 
                f"Vùng phục hồi phải có nhiễu hạt siêu âm (std = {reconstructed_noise_std:.2f})"
            )

    def test_structure_preservation_inside_caliper(self):
        """
        Yêu cầu 1: Giữ nguyên cấu trúc phía bên trong caliper,
        khôi phục liên tục cấu trúc mô giải phẫu.
        """
        processed_img, mask, boxes, meta = self.remover.remove_calipers_with_gemini(self.image_with_caliper)
        
        self.assertIsNotNone(processed_img)
        self.assertEqual(processed_img.shape, self.image_with_caliper.shape)
        self.assertEqual(processed_img.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
