import os, subprocess, tempfile, sys
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import cv2
import numpy as np
import torch
from highlight.process_images import highlight_and_extract_all_boxes

ROOT = Path(__file__).parent
LAMA_DIR = Path(os.getenv("LAMA_DIR", ROOT / "lama"))
MODEL_DIR = Path(os.getenv("LAMA_MODEL", ROOT / "big-lama"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
app = FastAPI(title="Caliper Cleanroom / LaMa")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"lama": (LAMA_DIR / "bin" / "predict.py").exists(), "model": MODEL_DIR.exists()}

def make_detector_mask(image_path: Path, mask_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise HTTPException(400, "Không đọc được ảnh để tìm dấu đánh dấu")
    _, boxes = highlight_and_extract_all_boxes(image.copy(), str(ROOT / "highlight" / "templates"), threshold=0.60)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for box in boxes:
        # Bao phủ toàn bộ vùng template/bounding box để LaMa xóa sạch dấu.
        cv2.rectangle(mask, (box['xmin'], box['ymin']), (box['xmax'], box['ymax']), 255, -1)
    cv2.imwrite(str(mask_path), mask)
    return len(boxes)

@app.post("/inpaint")
async def inpaint(image: UploadFile = File(...), mask: UploadFile | None = File(default=None)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "Ảnh không hợp lệ")
    with tempfile.TemporaryDirectory(prefix="caliper-lama-") as tmp:
        work = Path(tmp); inp = work / "input"; out = work / "output"; inp.mkdir()
        image_path = inp / "image.png"
        mask_path = inp / "image_mask001.png"
        image_path.write_bytes(await image.read())
        if mask is not None:
            mask_path.write_bytes(await mask.read())
            supplied = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if supplied is None or not np.any(supplied > 0):
                count = make_detector_mask(image_path, mask_path)
            else:
                count = -1
        else:
            count = make_detector_mask(image_path, mask_path)
            if count == 0:
                raise HTTPException(422, "Không tìm thấy dấu đánh dấu bằng template matching")
        predictor = LAMA_DIR / "bin" / "predict.py"
        if not predictor.exists() or not MODEL_DIR.exists():
            raise HTTPException(503, "Chưa tìm thấy LaMa: cần ./lama và ./big-lama")
        try:
            run = subprocess.run([sys.executable, str(predictor), f"model.path={MODEL_DIR.resolve()}", f"indir={inp.resolve()}", f"outdir={out.resolve()}", f"device={DEVICE}"], cwd=LAMA_DIR, check=True, capture_output=True, text=True, timeout=300)
        except subprocess.CalledProcessError as exc:
            raise HTTPException(500, exc.stderr[-1200:])
        candidates = list(out.rglob("*.png")) + list(out.rglob("*.jpg")) + list(out.rglob("*.jpeg"))
        # LaMa names the result after the mask file (e.g. image_mask001.png).
        result = candidates[0] if candidates else None
        if result is None:
            raise HTTPException(500, "LaMa không tạo được ảnh kết quả. Output: " + run.stdout[-800:] + run.stderr[-800:])
        return Response(content=result.read_bytes(), media_type="image/png", headers={"Content-Disposition": "attachment; filename=caliper-cleaned.png"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
