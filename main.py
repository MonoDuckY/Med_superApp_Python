from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
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

@app.get("/mockup", response_class=HTMLResponse)
async def get_mockup():
    """
    Giao diện mockup để upload và test trực tiếp AI Pipeline.
    """
    import os
    file_path = os.path.join(os.path.dirname(__file__), "mockup.html")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

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
import base64
import cv2
import numpy as np
from core_pipeline.gemini_caliper_remover import gemini_caliper_remover, GeminiUltrasoundCaliperRemover

from typing import Optional

@app.post("/api/v1/ai/remove-calipers-gemini")
async def remove_calipers_gemini_endpoint(
    file: UploadFile = File(...),
    api_key: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    model: Optional[str] = Form(default="gemini-3.1-pro")
):
    """
    Xóa toàn bộ Caliper trên ảnh siêu âm bằng LLM Gemini 3.1 Pro Multimodal Vision:
    - Giữ nguyên cấu trúc phía bên trong caliper
    - Giữ lại toàn bộ 100% cấu trúc các vùng khác (MSE = 0.0)
    - Giữ nguyên đặc trưng nhiễu hạt siêu âm (Speckle Noise Matching)
    - Giữ nguyên texture & màu sắc đơn sắc (Chống ngả màu RGB)
    """
    if file.content_type and not file.content_type.startswith("image/"):
        return {
            "success": False,
            "message": "File upload phải là hình ảnh.",
            "data": None,
            "errorCode": "INVALID_FILE_TYPE"
        }

    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return {
                "success": False,
                "message": "Không thể giải mã file ảnh.",
                "data": None,
                "errorCode": "DECODE_ERROR"
            }

        # Làm sạch api_key, prompt & model
        clean_api_key = api_key.strip() if api_key and api_key.strip() else None
        clean_prompt = prompt.strip() if prompt and prompt.strip() else None
        clean_model = model.strip() if model and model.strip() else "gemini-3.1-pro"

        # Thực thi Gemini Caliper Removal trong thread riêng
        processed_img, mask, boxes, meta = await asyncio.to_thread(
            gemini_caliper_remover.remove_calipers_with_gemini,
            image,
            clean_api_key,
            clean_prompt,
            clean_model
        )



        # Base64 encoding
        _, orig_buf = cv2.imencode('.jpg', image)
        _, proc_buf = cv2.imencode('.jpg', processed_img)
        _, mask_buf = cv2.imencode('.png', mask)

        orig_b64 = f"data:image/jpeg;base64,{base64.b64encode(orig_buf).decode('utf-8')}"
        proc_b64 = f"data:image/jpeg;base64,{base64.b64encode(proc_buf).decode('utf-8')}"
        mask_b64 = f"data:image/png;base64,{base64.b64encode(mask_buf).decode('utf-8')}"

        return {
            "success": True,
            "message": "Xóa caliper bằng LLM Gemini thành công.",
            "data": {
                "original_image_base64": orig_b64,
                "processed_image_base64": proc_b64,
                "caliper_mask_base64": mask_b64,
                "detected_boxes": boxes,
                "metrics": meta
            },
            "errorCode": None
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Lỗi xử lý caliper với Gemini: {str(e)}",
            "data": None,
            "errorCode": "GEMINI_CALIPER_ERROR"
        }


@app.post("/api/v1/ai/process-single")
async def process_single_image(
    file: UploadFile = File(...),
    options: str = Form("{}")
):
    """
    Endpoint tương thích với Spring Boot Backend (AiServiceClient).
    """
    if not file.content_type.startswith("image/"):
        return {
            "success": False,
            "message": "File upload phải là hình ảnh.",
            "data": None,
            "errorCode": "INVALID_FILE_TYPE"
        }

    try:
        opts = json.loads(options) if options else {}
    except Exception:
        opts = {}

    try:
        image_bytes = await file.read()
        patient_id = opts.get("patient_id", "Unknown")
        use_gemini = opts.get("use_gemini", True)
        api_key = opts.get("gemini_api_key", None)

        result_data = await asyncio.to_thread(
            run_ultrasound_pipeline,
            image_bytes,
            patient_id,
            use_gemini,
            api_key
        )

        return {
            "success": True,
            "message": "Xử lý ảnh siêu âm thành công.",
            "data": result_data,
            "errorCode": None
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Lỗi trong quá trình xử lý: {str(e)}",
            "data": None,
            "errorCode": "PROCESS_ERROR"
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
