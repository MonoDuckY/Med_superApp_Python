from fastapi import FastAPI

app = FastAPI(
    title="Medical AI Service",
    description="Microservice xử lý ảnh và chẩn đoán",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "UP", "message": "Python AI Service is ready!"}

if __name__ == "__main__":
    import uvicorn
    # Chạy server ở cổng 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
