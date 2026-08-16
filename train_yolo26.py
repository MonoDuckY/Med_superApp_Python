from ultralytics import YOLO
import torch

def train_model():
    # Check GPU availability
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, falling back to CPU.")
        device = 'cpu'
        
    # Initialize YOLOv26 Medium classification model
    print("Initializing yolo26m-cls model...")
    model = YOLO('yolo26m-cls.pt')
    
    # Train the model
    print(f"Starting training on data='dataset_cls' for 100 epochs on device={device}...")
    results = model.train(
        data='dataset_cls',
        epochs=100,
        imgsz=640,
        device=device,
        workers=4,     # optimize data loading
        batch=8        # reduced batch size to fit in 4GB VRAM
    )
    
    print("Training complete!")
    print(results)

if __name__ == '__main__':
    train_model()
