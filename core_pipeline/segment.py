import logging

logger = logging.getLogger(__name__)

class MedSAM_InferenceModel:
    def __init__(self):
        logger.warning("CẢNH BÁO: Đang sử dụng Mock MedSAM_InferenceModel do file segment.py gốc bị thiếu trong source code!")
        
    def predict(self, image, bbox):
        # Trả về kết quả rỗng (mock)
        return {"annotations": []}
