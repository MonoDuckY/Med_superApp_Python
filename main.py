from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Header
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import io
import json
import os
import time
import uuid
from pathlib import Path
import logging
import sys

# Cấu hình Logging toàn cục
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
from pydantic import BaseModel

from geometry import (
    CaliperPair,
    Point,
    lesion_region_from_calipers,
    suggest_calipers_from_bbox,
    suggest_calipers_from_mask
)

from core_pipeline.pipeline import run_ultrasound_pipeline, batch_process_dataset
from core_pipeline.training import train_yolo_resnet, job_statuses

load_dotenv()
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.15"))

_model = None
_current_repo_id = None
_current_filename = None

app = FastAPI(
    title="Medical AI Service",
    description="Microservice xử lý ảnh và chẩn đoán",
    version="1.0.0"
)

# Cấu hình CORS (cho phép frontend gọi API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {
        "status": "UP", 
        "message": "Python AI Service is ready!",
        "model_loaded": _model is not None
    }

@app.post("/api/v1/ai/analyze-ultrasound")
async def analyze_ultrasound(
    file: UploadFile = File(...),
    patient_id: str = Form("Unknown")
):
    """
    Nhận ảnh siêu âm và đưa qua AI Pipeline (Preprocess -> Enhance -> Segment).
    """
    if not file.content_type.startswith("image/"):
        return {
            "success": False,
            "message": "File upload phải là hình ảnh.",
            "data": None,
            "errorCode": "INVALID_FILE_TYPE"
        }

    try:
        image_bytes = await file.read()
        
        # Chạy pipeline AI trong thread pool để không block luồng chính của FastAPI
        result_data = await asyncio.to_thread(
            run_ultrasound_pipeline, 
            image_bytes, 
            patient_id
        )

        return {
            "success": True,
            "message": "Phân tích ảnh siêu âm thành công.",
            "data": result_data,
            "errorCode": None
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Lỗi trong quá trình xử lý ảnh: {str(e)}",
            "data": None,
            "errorCode": "PIPELINE_ERROR"
        }

def load_ai_model(repo_id: str, filename: str):
    global _model, _current_repo_id, _current_filename
    print(f"Đang tải model từ HF: {repo_id}/{filename} ...")
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("Chưa cài ultralytics. Chạy: pip install ultralytics") from e
    
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=HF_TOKEN
    )
    _model = YOLO(model_path)
    _current_repo_id = repo_id
    _current_filename = filename
    print("Model đã sẵn sàng!")

def get_model():
    """Lazy-load model — chỉ nạp 1 lần, dùng lại cho mọi request."""
    global _model
    if _model is None:
        load_ai_model(_current_repo_id, _current_filename)
    return _model

class ReloadModelRequest(BaseModel):
    repo_id: str
    filename: str

@app.post("/api/reload-model")
async def reload_model(req: ReloadModelRequest, authorization: str | None = Header(default=None)):
    """
    Cập nhật model mới theo repo_id và filename (từ Admin qua C# Backend).
    """
    try:
        expected_token = os.environ.get("WEBHOOK_TOKEN")
        if not expected_token:
            raise HTTPException(status_code=500, detail="WEBHOOK_TOKEN is not configured on the server")
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        
        token = authorization.split(" ")[1]
        if token != expected_token:
            raise HTTPException(status_code=403, detail="Invalid token")

        load_ai_model(req.repo_id, req.filename)
        return {"status": "success", "message": f"Đã chuyển sang model {req.filename}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PointIn(BaseModel):
    x: float
    y: float

@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
    repo_id: str | None = Form(default=None),
    filename: str | None = Form(default=None)
):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "File ảnh không hợp lệ.")

    img_w, img_h = image.size
    session_id = str(uuid.uuid4())

    if repo_id and filename:
        if _model is None or _current_repo_id != repo_id or _current_filename != filename:
            load_ai_model(repo_id, filename)

    IOU_THRESHOLD = float(os.environ.get("IOU_THRESHOLD", "0.1"))
    model = get_model()
    results = model.predict(np.array(image), conf=CONF_THRESHOLD, verbose=False)
    r = results[0]

    def calculate_iou(box1, box2):
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    raw_detections = []
    for i, box in enumerate(r.boxes):
        raw_detections.append({
            "orig_idx": i,
            "bbox": [float(box.xyxy[0][0]), float(box.xyxy[0][1]), float(box.xyxy[0][2]), float(box.xyxy[0][3])],
            "conf": float(box.conf[0]),
            "cls_id": int(box.cls[0])
        })

    raw_detections.sort(key=lambda x: x["conf"], reverse=True)

    kept_detections = []
    for d in raw_detections:
        keep = True
        for kd in kept_detections:
            if calculate_iou(d["bbox"], kd["bbox"]) > IOU_THRESHOLD:
                keep = False
                break
        if keep:
            kept_detections.append(d)

    detections = []
    has_mask = getattr(r, "masks", None) is not None
    for d in kept_detections:
        i = d["orig_idx"]
        x1, y1, x2, y2 = d["bbox"]
        conf = d["conf"]
        cls_id = d["cls_id"]

        if has_mask:
            mask_arr = r.masks.data[i].cpu().numpy()
            if mask_arr.shape != (img_h, img_w):
                import cv2
                mask_arr = cv2.resize(mask_arr, (img_w, img_h))
            pair_a, pair_b = suggest_calipers_from_mask(mask_arr)
        else:
            pair_a, pair_b = suggest_calipers_from_bbox(x1, y1, x2, y2)

        nx1, ny1, nx2, ny2 = x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h
        detections.append({
            "confidence": conf,
            "class_id": cls_id,
            "bbox": {"xmin": nx1, "ymin": ny1, "xmax": nx2, "ymax": ny2},
            "suggested_calipers": {
                "pair_a": [pair_a.p1.as_tuple(), pair_a.p2.as_tuple()],
                "pair_b": [pair_b.p1.as_tuple(), pair_b.p2.as_tuple()],
            },
        })

    return {
        "session_id": session_id,
        "image_width": img_w,
        "image_height": img_h,
        "detections": detections,
    }

@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Tải về file ZIP kết quả đã được xử lý.
    """
    if not filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Chỉ cho phép tải file ZIP")
        
    outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
    file_path = os.path.join(outputs_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Không tìm thấy file kết quả")
        
    return FileResponse(path=file_path, filename=filename, media_type="application/zip")

@app.post("/api/v1/ai/research/preprocess-dataset")
async def preprocess_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    webhook_url: str = Form(""),
    options: str = Form("{}")
):
    """
    Tiền xử lý hàng loạt ảnh siêu âm từ file zip (UC-23).
    """
    if not file.filename.endswith(".zip"):
        return {"success": False, "message": "File phải là định dạng .zip", "data": None, "errorCode": "INVALID_FILE"}
        
    job_id = str(uuid.uuid4())
    zip_bytes = await file.read()
    
    try:
        opts = json.loads(options)
    except Exception:
        opts = {}
        
    background_tasks.add_task(batch_process_dataset, job_id, zip_bytes, webhook_url, opts)
    
    return {
        "success": True,
        "message": "Đã tiếp nhận yêu cầu tiền xử lý dataset.",
        "data": {"job_id": job_id},
        "errorCode": None
    }

@app.post("/api/v1/ai/research/train-model")
async def train_model(
    background_tasks: BackgroundTasks,
    dataset: UploadFile = File(...),
    model_type: str = Form("yolov8_resnet"),
    epochs: int = Form(50),
    webhook_url: str = Form("")
):
    """
    Huấn luyện AI Model (UC-24).
    """
    if not dataset.filename.endswith(".zip"):
        return {"success": False, "message": "Dataset phải là định dạng .zip", "data": None, "errorCode": "INVALID_FILE"}
        
    job_id = str(uuid.uuid4())
    
    # Save the dataset to a temporary/outputs directory
    outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    dataset_path = os.path.join(outputs_dir, f"dataset_{job_id}.zip")
    
    with open(dataset_path, "wb") as f:
        f.write(await dataset.read())
    
    background_tasks.add_task(train_yolo_resnet, job_id, model_type, epochs, webhook_url, dataset_path)
    
    return {
        "success": True,
        "message": "Đã nhận dataset và bắt đầu quá trình huấn luyện mô hình.",
        "data": {"job_id": job_id},
        "errorCode": None
    }

@app.get("/api/v1/ai/research/job-status/{job_id}")
async def get_job_status(job_id: str):
    """
    Lấy trạng thái hiện tại của một job (VD: training progress).
    """
    if job_id not in job_statuses:
        raise HTTPException(status_code=404, detail="Job ID không tồn tại hoặc đã bị xóa khỏi bộ nhớ.")
        
    return {
        "success": True,
        "data": job_statuses[job_id],
        "errorCode": None
    }

@app.get("/api/v1/ai/research/job-statuses")
async def get_all_job_statuses():
    """
    Lấy trạng thái của tất cả các jobs đang lưu trong bộ nhớ.
    """
    return {
        "success": True,
        "data": job_statuses,
        "errorCode": None
    }

if __name__ == "__main__":
    import uvicorn
    # Chạy server ở cổng 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
