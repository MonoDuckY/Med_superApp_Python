from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import numpy as np
import os
import asyncio
import tempfile
import zipfile
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor

from inpaint import inpaint_calipers

# Dictionary to store batch job statuses
batch_jobs = {}

app = FastAPI(title="LaMa Caliper Removal Mockup")

# Serve the static HTML frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

# Use a thread pool for batch processing to limit concurrency and memory usage
MAX_CONCURRENT_IMAGES = 4
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_IMAGES)

def process_single_image(input_path: str, output_path: str, label_path: str = None):
    """Reads, processes, and saves a single image, optionally saving caliper boxes as YOLO labels."""
    try:
        image = cv2.imread(input_path)
        if image is None:
            return False
            
        result_img, _, boxes = inpaint_calipers(image, TEMPLATES_DIR)
        cv2.imwrite(output_path, result_img)
        
        if label_path:
            if boxes:
                # Calculate Union Bounding Box
                h, w = image.shape[:2]
                xmin = min(b['xmin'] for b in boxes)
                ymin = min(b['ymin'] for b in boxes)
                xmax = max(b['xmax'] for b in boxes)
                ymax = max(b['ymax'] for b in boxes)
                
                # YOLO format: class x_center y_center width height
                x_center = ((xmin + xmax) / 2) / w
                y_center = ((ymin + ymax) / 2) / h
                width = (xmax - xmin) / w
                height = (ymax - ymin) / h
                
                with open(label_path, "w") as f:
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            else:
                # Create empty file if no calipers detected (background image)
                open(label_path, 'a').close()
                
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def cleanup_temp_dir(temp_dir: str):
    """Remove temporary directory after response is sent."""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Failed to cleanup temp dir {temp_dir}: {e}")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

def process_batch_background(job_id: str, zip_path: str, extract_dir: str, output_dir: str, result_zip_path: str, temp_dir: str):
    try:
        batch_jobs[job_id]["status"] = "processing"
        batch_jobs[job_id]["progress"] = "Extracting files..."
        
        # 2. Extract zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # 3. Find all images and set up output structure
        tasks = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        
        image_files = []
        has_data_yaml = False
        
        # Check if data.yaml exists in the input zip
        for root, dirs, files in os.walk(extract_dir):
            if any(f.lower() == "data.yaml" for f in files):
                has_data_yaml = True
                break
                
        # Define YOLO directories if restructuring
        yolo_img_dir = os.path.join(output_dir, "images", "train")
        yolo_lbl_dir = os.path.join(output_dir, "labels", "train")
        if not has_data_yaml:
            os.makedirs(yolo_img_dir, exist_ok=True)
            os.makedirs(yolo_lbl_dir, exist_ok=True)
        
        for root, dirs, files in os.walk(extract_dir):
            for filename in files:
                input_path = os.path.join(root, filename)
                
                is_image = filename.lower().endswith(valid_extensions) and not filename.startswith('._')
                
                if has_data_yaml:
                    # Preserve original structure
                    rel_path = os.path.relpath(root, extract_dir)
                    target_dir = os.path.join(output_dir, rel_path)
                    os.makedirs(target_dir, exist_ok=True)
                    output_path = os.path.join(target_dir, filename)
                    
                    if is_image:
                        image_files.append((input_path, output_path))
                    else:
                        shutil.copy2(input_path, output_path)
                else:
                    # Force restructure into YOLO format
                    if is_image:
                        output_path = os.path.join(yolo_img_dir, filename)
                        base_name = os.path.splitext(filename)[0]
                        lbl_path = os.path.join(yolo_lbl_dir, f"{base_name}.txt")
                        image_files.append((input_path, output_path, lbl_path))
                    else:
                        # Copy other files to root
                        output_path = os.path.join(output_dir, filename)
                        shutil.copy2(input_path, output_path)
                        
        if not has_data_yaml:
            yaml_content = "path: .\ntrain: images/train\nval: images/train\n\nnc: 1\nnames: ['lesion']\n"
            with open(os.path.join(output_dir, "data.yaml"), "w") as f:
                f.write(yaml_content)
                    
        total_images = len(image_files)
        if total_images == 0:
            batch_jobs[job_id]["status"] = "failed"
            batch_jobs[job_id]["progress"] = "No valid images found."
            cleanup_temp_dir(temp_dir)
            return
            
        batch_jobs[job_id]["progress"] = f"Processing 0/{total_images} images..."
        
        # Process sequentially in background to update progress and avoid overloading CPU
        processed = 0
        for item in image_files:
            if len(item) == 3:
                process_single_image(item[0], item[1], item[2])
            else:
                process_single_image(item[0], item[1])
            processed += 1
            batch_jobs[job_id]["progress"] = f"Processing {processed}/{total_images} images..."

        batch_jobs[job_id]["progress"] = "Zipping results..."
        # 5. Zip the output directory
        with zipfile.ZipFile(result_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)

        batch_jobs[job_id]["status"] = "completed"
        batch_jobs[job_id]["progress"] = "Ready for download"
        batch_jobs[job_id]["result_path"] = result_zip_path
        # We DO NOT cleanup temp_dir here because we need it for download.
        # Cleanup will happen when download is triggered or via a cron task (not implemented here).

    except Exception as e:
        batch_jobs[job_id]["status"] = "failed"
        batch_jobs[job_id]["progress"] = str(e)
        cleanup_temp_dir(temp_dir)

@app.post("/api/inpaint")
async def process_batch(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only .zip files are supported for batch processing")

    job_id = str(uuid.uuid4())
    batch_jobs[job_id] = {
        "status": "pending",
        "progress": "Uploading...",
        "result_path": None
    }

    # Create temporary directories for processing
    temp_dir = tempfile.mkdtemp()
    extract_dir = os.path.join(temp_dir, "input")
    output_dir = os.path.join(temp_dir, "output")
    zip_path = os.path.join(temp_dir, "upload.zip")
    result_zip_path = os.path.join(temp_dir, "result.zip")
    
    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. Save uploaded zip
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Spawn background task
        background_tasks.add_task(
            process_batch_background, 
            job_id, zip_path, extract_dir, output_dir, result_zip_path, temp_dir
        )
        
        return {"job_id": job_id, "status": "processing"}

    except Exception as e:
        cleanup_temp_dir(temp_dir)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/api/inpaint_status/{job_id}")
async def get_inpaint_status(job_id: str):
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return batch_jobs[job_id]

@app.get("/api/inpaint_download/{job_id}")
async def download_inpaint_result(background_tasks: BackgroundTasks, job_id: str):
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job_info = batch_jobs[job_id]
    if job_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed yet")
        
    result_path = job_info["result_path"]
    
    # Optional: Schedule cleanup after response
    temp_dir = os.path.dirname(result_path)
    background_tasks.add_task(cleanup_temp_dir, temp_dir)
    
    return FileResponse(
        path=result_path, 
        filename="processed_images.zip",
        media_type="application/zip"
    )

@app.post("/api/inpaint_single")
async def process_single(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
        
    try:
        loop = asyncio.get_event_loop()
        def do_inpaint():
            result_img, _ = inpaint_calipers(image, TEMPLATES_DIR)
            return result_img
            
        result_img = await loop.run_in_executor(executor, do_inpaint)
        
        _, encoded_img = cv2.imencode('.jpg', result_img)
        return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
