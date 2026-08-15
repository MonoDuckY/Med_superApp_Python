import time
import logging

logger = logging.getLogger(__name__)

def evaluate_yolo_model(model_path: str, dataset_path: str = None) -> dict:
    """
    Evaluates a trained YOLO model against a dataset using actual YOLO validation.
    Returns real metrics.
    """
    logger.info(f"Đang bắt đầu đánh giá model: {model_path} ...")
    if dataset_path:
        logger.info(f"Đánh giá trên tập test từ: {dataset_path}")
        
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Thư viện ultralytics chưa được cài đặt.")
        return {}

    try:
        model = YOLO(model_path)
        # Chạy quá trình validation thực tế
        metrics_obj = model.val(data=dataset_path, verbose=False)
        
        logger.info("Quá trình đánh giá hoàn tất.")
        
        # Trích xuất các chỉ số thực tế
        # mAP50, mAP50-95, precision mean, recall mean, f1 mean
        p_mean = float(metrics_obj.box.mp) if hasattr(metrics_obj.box, 'mp') else 0.0
        r_mean = float(metrics_obj.box.mr) if hasattr(metrics_obj.box, 'mr') else 0.0
        map50 = float(metrics_obj.box.map50) if hasattr(metrics_obj.box, 'map50') else 0.0
        map_val = float(metrics_obj.box.map) if hasattr(metrics_obj.box, 'map') else 0.0
        
        # F1 score xấp xỉ từ Precision và Recall mean
        f1_mean = 2 * (p_mean * r_mean) / (p_mean + r_mean + 1e-16)
        
        # Lấy tốc độ inference (ms)
        speed = sum(metrics_obj.speed.values()) if hasattr(metrics_obj, 'speed') else 0.0
        
        return {
            "mAP50": round(map50, 4),
            "mAP50-95": round(map_val, 4),
            "precision": round(p_mean, 4),
            "recall": round(r_mean, 4),
            "f1_score": round(f1_mean, 4),
            "inference_speed_ms": round(speed, 2)
        }
    except Exception as e:
        logger.error(f"Lỗi trong quá trình đánh giá: {e}")
        return {
            "error": str(e)
        }
