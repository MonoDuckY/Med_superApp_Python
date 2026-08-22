# Med SuperApp — AI Inpainting & Medical Image Processing Service

Dịch vụ AI Microservice chuyên xử lý ảnh y tế và loại bỏ dấu thước đo siêu âm (Caliper Removal) dựa trên Computer Vision (Template Matching) và Deep Learning (LaMa Inpainting).

---

## 🛠️ Tổng Quan Công Nghệ & Kiến Trúc

- **Framework:** FastAPI (Python 3.10)
- **Deep Learning / Inpainting:** LaMa (Large Mask Inpainting) trên nền tảng PyTorch
- **Computer Vision:** OpenCV (Phát hiện caliper qua template matching đa tỷ lệ), NumPy
- **Server:** Uvicorn ASGI Server
- **Khả năng tăng tốc:** Hỗ trợ tự động NVIDIA CUDA (GPU) hoặc fallback về CPU

### Pipeline Xử Lý
```text
Ảnh siêu âm gốc 
    │
    ▼
[Template Matching (OpenCV)] ──(Tự động phát hiện dấu caliper)──► [Tạo Bounding Mask]
    │                                                                   │
    └───────────────────────┬───────────────────────────────────────────┘
                            ▼
               [LaMa Inpainting (PyTorch)]
                            │
                            ▼
               [Ảnh siêu âm sạch dấu thước đo]
```

---

## 📁 Cấu Trúc Thư Mục

```
Med_superApp_Python/
├── removeByAI/                ← Thư mục chính của dịch vụ AI
│   ├── highlight/             ← Module OpenCV Template Matching
│   │   ├── templates/         ← Bộ template mẫu các dấu caliper (+, x, chấm, vạch)
│   │   └── process_images.py  ← Logic phát hiện và trích xuất bounding box
│   ├── lama/                  ← Kiến trúc mô hình LaMa Inpainting (PyTorch)
│   ├── big-lama/              ← Trọng số mô hình đã huấn luyện (tải tự động)
│   ├── server.py              ← FastAPI Application & API endpoints
│   ├── process_batch.py       ← Xử lý hàng loạt ảnh từ thư mục
│   ├── setup_project.bat      ← Script tự động cài đặt môi trường & tải weights
│   ├── start_backend.bat      ← Script khởi chạy FastAPI server
│   ├── stop_backend.bat       ← Script dừng FastAPI server
│   ├── run_test.bat           ← Script chạy batch test ảnh
│   ├── requirements-lama.txt  ← Danh sách thư viện Python cần thiết
│   ├── index.html & app.js    ← Giao diện Web kiểm thử nhanh
│   └── styles.css             ← Giao diện UI test
└── README.md
```

---

## 🚀 Cài Đặt & Khởi Chạy

### 1. Yêu Cầu Hệ Thống
- **Hệ điều hành:** Windows 10/11 hoặc Linux
- **Python:** Phiên bản `3.10.x`
- **Phần cứng:** Khuyến nghị có GPU NVIDIA (VRAM >= 4GB) kèm CUDA để xử lý thời gian thực (< 1 giây/ảnh); CPU vẫn hoạt động bình thường nhưng thời gian xử lý khoảng 3-8 giây/ảnh.

---

### 2. Cài Đặt Tự Động (Khuyến nghị trên Windows)

1. Mở PowerShell hoặc Command Prompt tại thư mục `removeByAI/`:
   ```cmd
   cd removeByAI
   setup_project.bat
   ```
2. Script sẽ tự động:
   - Khởi tạo môi trường ảo Python (`.venv`).
   - Cài đặt PyTorch và toàn bộ dependencies từ `requirements-lama.txt`.
   - Tải bộ trọng số mô hình `big-lama` về thư mục local.
   - Kiểm tra khả dụng của GPU CUDA.

---

### 3. Cài Đặt Thủ Công

Nếu muốn cài đặt thủ công:
```bash
cd removeByAI

# 1. Tạo và kích hoạt virtual environment
python -m venv .venv
# Trên Windows:
.\.venv\Scripts\activate
# Trên Linux/macOS:
# source .venv/bin/activate

# 2. Cài đặt dependencies
pip install --upgrade pip
pip install -r requirements-lama.txt
```

---

### 4. Khởi Chạy Dịch Vụ

#### Cách 1: Dùng script có sẵn
```cmd
cd removeByAI
start_backend.bat
```

#### Cách 2: Chạy trực tiếp qua Uvicorn
```bash
cd removeByAI
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

- **Base URL:** `http://127.0.0.1:8000`
- **Kiểm tra trạng thái:** `http://127.0.0.1:8000/health`
- **Tài liệu Swagger UI:** `http://127.0.0.1:8000/docs`

---

## 📡 Đặc Tả API

### 1. Health Check
- **Endpoint:** `GET /health`
- **Mô tả:** Kiểm tra sự sẵn sàng của mã nguồn LaMa và bộ trọng số `big-lama`.
- **Response:**
  ```json
  {
    "lama": true,
    "model": true
  }
  ```

### 2. Xóa Dấu Thước Đo (Inpaint Image)
- **Endpoint:** `POST /inpaint`
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `image` *(bắt buộc)*: File ảnh siêu âm (`.jpg`, `.png`, `.webp`, `.tiff`).
  - `mask` *(tùy chọn)*: File ảnh mask nhị phân (đen/trắng). Nếu không gửi mask, hệ thống sẽ tự động dùng OpenCV Template Matching để nhận diện vị trí các caliper và sinh mask tự động.
- **Response:** File ảnh kết quả dạng binary `image/png` (đã xóa dấu caliper và tái tạo texture nền).
- **Mã lỗi thường gặp:**
  - `400 BAD REQUEST`: Định dạng ảnh không hợp lệ hoặc không đọc được.
  - `422 UNPROCESSABLE ENTITY`: Không phát hiện thấy dấu caliper trên ảnh.
  - `503 SERVICE UNAVAILABLE`: Chưa cấu hình model `big-lama` hoặc mã nguồn LaMa.

---

## 🧪 Công Cụ Kiểm Thử Nhanh

### 1. Giao diện Web Tester
Sau khi khởi chạy backend (`start_backend.bat`), mở trực tiếp file `removeByAI/index.html` trên trình duyệt:
- Tải ảnh siêu âm lên.
- Có thể dùng cọ vẽ thủ công vùng muốn xóa hoặc để AI tự động nhận diện.
- Bấm **XỬ LÝ ẢNH** và so sánh trực quan ảnh trước/sau.

### 2. Xử Lý Hàng Loạt (Batch Processing)
1. Đặt tất cả ảnh cần xử lý vào thư mục `removeByAI/input/`.
2. Chạy script:
   ```cmd
   cd removeByAI
   run_test.bat
   ```
3. Xem ảnh kết quả được xuất tự động tại `removeByAI/output/`.

---

## ⚠️ Lưu Ý Về Bảo Mật & Y Khoa
1. **Bảo Mật Dữ Liệu:** Toàn bộ quá trình xử lý ảnh diễn ra hoàn toàn nội bộ/cục bộ (On-premise / Local Server), không chuyển dữ liệu bệnh nhân ra ngoài internet.
2. **Khuyến Cáo Y Khoa:** Mô hình AI Inpainting thực hiện tái tạo kết cấu vùng mô bị che bởi thước đo. Kết quả chỉ phục vụ hỗ trợ quan sát, bác sĩ luôn đối chiếu với ảnh gốc trong quá trình chẩn đoán lâm sàng.
