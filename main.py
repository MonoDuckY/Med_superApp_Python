from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio

from core_pipeline.pipeline import run_ultrasound_pipeline, batch_process_dataset
from core_pipeline.training import train_yolo_resnet

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
    return {"status": "UP", "message": "Python AI Service is ready!"}

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

import uuid
import json
import os

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
    model_type: str = Form("yolov8_resnet"),
    epochs: int = Form(50),
    webhook_url: str = Form("")
):
    """
    Huấn luyện AI Model (UC-24).
    """
    job_id = str(uuid.uuid4())
    
    background_tasks.add_task(train_yolo_resnet, job_id, model_type, epochs, webhook_url)
    
    return {
        "success": True,
        "message": "Đã bắt đầu quá trình huấn luyện mô hình.",
        "data": {"job_id": job_id},
        "errorCode": None
    }


if __name__ == "__main__":
    import uvicorn
    # Chạy server ở cổng 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
