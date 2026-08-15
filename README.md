# Med Super App - Python AI Service

Đây là microservice chịu trách nhiệm tiền xử lý ảnh siêu âm (UC-23) và huấn luyện AI (UC-24) cho dự án Med Super App.
Được viết bằng FastAPI và sử dụng OpenCV để xử lý ảnh.

### 🚀 Hướng dẫn khởi chạy nhanh (Quick Start)

Để hệ thống hoạt động trơn tru (UC-23), bạn cần chạy đồng thời cả 3 component: **Python AI Service**, **Spring Boot Backend**, và **Next.js Frontend**. Hãy mở 3 cửa sổ Terminal (PowerShell) riêng biệt và chạy lần lượt các lệnh sau:

#### 1. Chạy AI Service (Python FastAPI - Port 8000)
```powershell
cd C:\Users\Acer\Documents\GitHub\Med_superApp_Python
.\venv\Scripts\activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# Hoặc: python main.py
```

#### 2. Chạy Backend (Spring Boot - Port 8080)
```powershell
cd C:\Users\Acer\Documents\GitHub\Med_superApp_backend
.\gradlew bootRun
```

#### 3. Chạy Frontend (Next.js - Port 3000)
```powershell
cd C:\Users\Acer\Documents\GitHub\Med_superApp_frontend\nextjs
npm run dev
```

Sau khi cả 3 server đều báo đã chạy thành công, hãy mở trình duyệt và truy cập vào giao diện Dataset Processing tại:
👉 **[http://localhost:3000/research](http://localhost:3000/research)**

---

### 🎨 Thử nghiệm trực tiếp với Mockup UI (Không cần Backend/Frontend)

Nếu bạn chỉ muốn kiểm tra nhanh thuật toán **Xóa thước đo (Caliper)** với tính năng giữ nguyên màu sắc, cấu trúc và đặc trưng nhiễu hạt (speckle noise) mà không cần bật toàn bộ hệ thống, bạn có thể dùng Mockup UI được tích hợp sẵn:

1. Chạy AI Service (Python FastAPI):
```powershell
cd C:\Users\Admin\Documents\GitHub\Med_superApp\Med_superApp_Python
.\venv\Scripts\activate
python main.py
```
2. Mở trình duyệt và truy cập:
👉 **[http://localhost:8000/mockup](http://localhost:8000/mockup)**

Giao diện sẽ cho phép bạn kéo thả/tải lên 1 file ảnh siêu âm và xem kết quả xử lý của thuật toán ngay lập tức.

---

### Cấu trúc dự án
- `main.py`: Các API endpoints của FastAPI (`/api/v1/ai/...`).
- `core_pipeline/`: Chứa mã nguồn lõi xử lý ảnh.
  - `pipeline.py`: Luồng chạy chính cho batch processing.
  - `preprocess.py`: Nhận diện Caliper (bằng Template Matching) và xoá chữ.
  - `enhance.py`: Chỉnh sáng tối, tương phản, độ nét, và bộ lọc SRAD.
  - `xml_exporter.py`: Trích xuất tọa độ Bounding Boxes ra chuẩn Pascal VOC XML.
  - `augment.py`: Module làm giàu dữ liệu (Data Augmentation) x4.
  - `templates/`: Thư mục chứa các mẫu hình ảnh dấu Caliper (+, x).
