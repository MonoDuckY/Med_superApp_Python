from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import numpy as np
import os
import asyncio
import tempfile
import zipfile
import shutil
from concurrent.futures import ThreadPoolExecutor

from inpaint import inpaint_calipers

app = FastAPI(title="LaMa Caliper Removal Mockup")

# Serve the static HTML frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

# Use a thread pool for batch processing to limit concurrency and memory usage
MAX_CONCURRENT_IMAGES = 4
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_IMAGES)

def process_single_image(input_path: str, output_path: str):
    """Reads, processes, and saves a single image."""
    try:
        image = cv2.imread(input_path)
        if image is None:
            return False
            
        result_img, _ = inpaint_calipers(image, TEMPLATES_DIR)
        cv2.imwrite(output_path, result_img)
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

@app.post("/api/inpaint")
async def process_batch(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only .zip files are supported for batch processing")

    # Create temporary directories for processing
    temp_dir = tempfile.mkdtemp()
    extract_dir = os.path.join(temp_dir, "input")
    output_dir = os.path.join(temp_dir, "output")
    zip_path = os.path.join(temp_dir, "upload.zip")
    result_zip_path = os.path.join(temp_dir, "result.zip")
    
    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Schedule cleanup after response
    background_tasks.add_task(cleanup_temp_dir, temp_dir)

    try:
        # 1. Save uploaded zip
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Extract zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # 3. Find all images and set up output structure
        tasks = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        
        for root, dirs, files in os.walk(extract_dir):
            for filename in files:
                if filename.lower().endswith(valid_extensions) and not filename.startswith('._'):
                    input_path = os.path.join(root, filename)
                    
                    # Mirror the directory structure in output_dir
                    rel_path = os.path.relpath(root, extract_dir)
                    target_dir = os.path.join(output_dir, rel_path)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    output_path = os.path.join(target_dir, filename)
                    
                    # Schedule task
                    loop = asyncio.get_event_loop()
                    tasks.append(
                        loop.run_in_executor(executor, process_single_image, input_path, output_path)
                    )

        if not tasks:
            raise HTTPException(status_code=400, detail="No valid image files found in the ZIP archive.")

        # 4. Run all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        if not any(results):
            raise HTTPException(status_code=500, detail="Failed to process any images in the ZIP archive.")

        # 5. Zip the output directory
        with zipfile.ZipFile(result_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)

        # 6. Return the zipped results
        return FileResponse(
            path=result_zip_path, 
            filename="processed_images.zip",
            media_type="application/zip"
        )

    except Exception as e:
        # Re-raise HTTP exceptions
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
