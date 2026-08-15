from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import uuid
import json
import os

from core_pipeline.pipeline import process_single_image, run_ultrasound_pipeline, batch_process_dataset

app = FastAPI(
    title="Ultrasound AI & Caliper Processing Service",
    description="Microservice tiền xử lý ảnh siêu âm, khử nhiễu SRAD, tô đỏ và trích xuất Caliper XML/JSON (UC21 & UC22)",
    version="2.0.0"
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
        "success": True,
        "message": "Python AI & Caliper Service is ready!",
        "data": {"status": "UP", "version": "2.0.0"},
        "errorCode": None
    }

@app.post("/api/v1/ai/process-single")
async def process_single(
    file: UploadFile = File(...),
    options: str = Form("{}")
):
    """
    Xử lý 1 ảnh siêu âm đơn cho Single Image Studio:
    - Khử nhiễu SRAD
    - Cắt Safe Area
    - Xóa chữ OCR
    - Nhận diện, tô đỏ Caliper và trích xuất XML / JSON.
    """
    try:
        # Kiểm tra định dạng an toàn
        ctype = (file.content_type or "").lower()
        fname = (file.filename or "").lower()
        valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".dcm")
        
        is_valid = (
            ctype.startswith("image/") or 
            ctype == "application/octet-stream" or 
            not ctype or 
            fname.endswith(valid_extensions)
        )

        if not is_valid:
            return {
                "success": False,
                "message": "Tệp tải lên phải là định dạng hình ảnh (.png, .jpg, .jpeg, .bmp).",
                "data": None,
                "errorCode": "INVALID_FILE_TYPE"
            }

        try:
            opts = json.loads(options) if options else {}
        except Exception:
            opts = {}

        image_bytes = await file.read()
        filename = file.filename or "ultrasound.jpg"

        # Chạy tác vụ xử lý ảnh trong threadpool
        result_data = await asyncio.to_thread(
            process_single_image,
            image_bytes,
            filename,
            opts
        )

        return {
            "success": True,
            "message": f"Xử lý ảnh thành công. Tìm thấy {result_data.get('caliper_count', 0)} dấu caliper.",
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


@app.post("/api/v1/ai/analyze-ultrasound")
async def analyze_ultrasound(
    file: UploadFile = File(...),
    patient_id: str = Form("Unknown"),
    options: str = Form("{}")
):
    """
    Tương thích ngược với các hàm gọi cũ.
    """
    return await process_single(file=file, options=options)

@app.post("/api/v1/ai/research/preprocess-dataset")
async def preprocess_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    webhook_url: str = Form(""),
    options: str = Form("{}")
):
    """
    Tiền xử lý hàng loạt ảnh siêu âm từ file zip (UC21 & UC22).
    """
    if not file.filename.endswith(".zip"):
        return {
            "success": False,
            "message": "File phải là định dạng nén .zip chứa ảnh siêu âm.",
            "data": None,
            "errorCode": "INVALID_FILE_FORMAT"
        }

    job_id = str(uuid.uuid4())
    zip_bytes = await file.read()

    try:
        opts = json.loads(options) if options else {}
    except Exception:
        opts = {}

    background_tasks.add_task(batch_process_dataset, job_id, zip_bytes, webhook_url, opts)

    return {
        "success": True,
        "message": "Đã tiếp nhận yêu cầu tiền xử lý dataset.",
        "data": {"job_id": job_id},
        "errorCode": None
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
        raise HTTPException(status_code=404, detail="Không tìm thấy file kết quả hoặc file đã bị xóa")

    return FileResponse(path=file_path, filename=filename, media_type="application/zip")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

