# LaMa Caliper Removal Mockup

This is a standalone mockup designed to demonstrate the removal of caliper markers from ultrasound images using the **Resolution-robust Large Mask Inpainting (LaMa)** model.

## Features
- Isolated environment focusing only on Caliper Mask Detection and LaMa Inpainting.
- Reuses the existing template matching logic (`detect_calipers`) to create masks.
- Sleek, modern Web UI for easy upload and visualization of the results (Original, Mask, Inpainted).
- Uses `simple-lama-inpainting` which abstracts the PyTorch complexity of the LaMa model.

## Installation

Ensure you are using Python 3.9+ and have a virtual environment active.

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

*(Note: When you first run the server and process an image, `simple-lama-inpainting` will automatically download the pre-trained weights to your local machine).*

## Running the Mockup

1. Start the FastAPI server:
```bash
python main.py
```

2. Open your web browser and navigate to:
```
http://localhost:8080
```

3. Drag and drop (or select) an ultrasound image that contains calipers (like the `+` or `x` marks). Wait for the AI processing to complete, and you'll see the original image, the generated mask, and the final inpainted image without the markers.
