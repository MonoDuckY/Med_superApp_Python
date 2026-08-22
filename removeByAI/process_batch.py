import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from highlight.process_images import highlight_and_extract_all_boxes

ROOT = Path(__file__).parent
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
TEMPLATES = ROOT / "highlight" / "templates"
LAMA = ROOT / "lama"
MODEL = ROOT / "big-lama"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def create_mask(image_path: Path, mask_path: Path) -> int:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Không đọc được ảnh: {image_path.name}")
    _, boxes = highlight_and_extract_all_boxes(image.copy(), str(TEMPLATES), threshold=0.60)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for box in boxes:
        cv2.rectangle(mask, (box["xmin"], box["ymin"]), (box["xmax"], box["ymax"]), 255, -1)
    cv2.imwrite(str(mask_path), mask)
    return len(boxes)

def process(image_path: Path):
    with tempfile.TemporaryDirectory(prefix="lama-batch-") as temp:
        temp = Path(temp)
        indir, outdir = temp / "input", temp / "output"
        indir.mkdir()
        Image.open(image_path).convert("RGB").save(indir / "image.png")
        count = create_mask(image_path, indir / "image_mask001.png")
        if count == 0:
            print(f"[SKIP] {image_path.name}: không tìm thấy dấu")
            return
        env = os.environ.copy()
        env["PYTHONPATH"] = str(LAMA)
        command = [sys.executable, str(LAMA / "bin" / "predict.py"), f"model.path={MODEL.resolve()}", f"indir={indir.resolve()}", f"outdir={outdir.resolve()}", f"device={DEVICE}"]
        subprocess.run(command, cwd=LAMA, env=env, check=True, stdout=subprocess.DEVNULL)
        results = list(outdir.rglob("*.png"))
        if not results:
            raise RuntimeError("LaMa không tạo output")
        target = OUTPUT / f"{image_path.stem}.png"
        shutil.copy2(results[0], target)
        print(f"[OK] {image_path.name} -> {target.name} ({count} dấu)")

def main():
    INPUT.mkdir(exist_ok=True)
    OUTPUT.mkdir(exist_ok=True)
    images = sorted(p for p in INPUT.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not images:
        print(f"Không có ảnh trong: {INPUT}")
        return
    print(f"Tìm thấy {len(images)} ảnh. Bắt đầu xử lý...\n")
    for image in images:
        try:
            process(image)
        except Exception as exc:
            print(f"[ERROR] {image.name}: {exc}")
    print(f"\nHoàn tất. Kết quả nằm tại: {OUTPUT}")

if __name__ == "__main__":
    main()
