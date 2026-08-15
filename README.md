# Medical AI Service - LLM Gemini Ultrasound Caliper Remover 🔬

Microservice AI chuyên sâu xử lý ảnh siêu âm y tế, tích hợp **LLM Gemini Multimodal Vision** để định vị và xóa triệt để các loại caliper (dấu `+`, `x`, thước đo, vạch chấm khoảng cách) với độ chính xác cao mà vẫn bảo toàn hoàn hảo cấu trúc mô giải phẫu, đặc trưng nhiễu hạt (speckle noise) và không bị ngả màu RGB.

---

## 🌟 4 Tiêu Chuẩn Kỹ Thuật Đạt Được
1. **Xóa caliper, bảo toàn cấu trúc bên trong caliper**: Tách đúng nét vẽ caliper (Sub-pixel Stroke Mask) và khôi phục liên tục cấu trúc giải phẫu bên dưới nét vẽ theo đường đẳng sáng (Isophote Navier-Stokes/Telea Inpainting).
2. **Giữ nguyên 100% các vùng khác**: Áp dụng kỹ thuật ghép mask nghiêm ngặt (Strict Mask Compositing), cam kết sai số **$MSE = 0.0$** ngoài vùng caliper.
3. **Giữ nguyên đặc trưng nhiễu hạt siêu âm (Speckle Noise Matching)**: Đo lường mức nhiễu hạt ($\sigma_{speckle}$) và độ phản hồi âm ($\mu$) từ vành khuyên bao quanh để tái tạo hạt siêu âm tự nhiên, không bị vệt mờ bệt trơn nhẵn.
4. **Giữ nguyên texture, chống ngả màu RGB**: Ép chuẩn đồng nhất kênh màu ($R=G=B$) cho ảnh siêu âm đen trắng, khử triệt để hiện tượng lem màu từ caliper màu vàng/xanh/đỏ vào mô y tế.

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Thử Nhanh Với Giao Diện HTML

Bạn có thể chạy thử nghiệm toàn bộ tính năng trực tiếp trên máy cục bộ chỉ trong **3 bước đơn giản**:

### Bước 1: Cài đặt Môi trường Ảo (Virtual Environment)
Mở cửa sổ **PowerShell** hoặc **Terminal** tại thư mục dự án:
```powershell
cd C:\Users\Acer\Documents\GitHub\Med_superApp_Python

# Tạo môi trường ảo (nếu chưa có)
python -m venv venv

# Kích hoạt môi trường ảo
.\venv\Scripts\activate

# Cài đặt các thư viện phụ thuộc (bao gồm google-genai, opencv, fastapi)
pip install -r requirements.txt
```

---

### Bước 2: Cấu hình Gemini API Key (Tùy chọn)
Hệ thống hỗ trợ cả **Gemini Cloud API** lẫn **Local Fallback (Offline)**:
* **Cách 1 (Khuyên dùng)**: Tạo file `.env` tại thư mục gốc `Med_superApp_Python` với nội dung:
  ```env
  GEMINI_API_KEY=AIzaSyYourGeminiApiKeyHere
  ```
  *(Lấy API Key miễn phí tại [Google AI Studio](https://aistudio.google.com/))*
* **Cách 2**: Nhập trực tiếp API Key trên giao diện Web Mockup HTML.
* **Cách 3**: Để trống nếu muốn chạy chế độ **Offline Local Fallback** (sử dụng Multi-scale Template Matching + Morphological Detector).

---

### Bước 3: Khởi động Server & Mở Giao Diện Test HTML

Chạy lệnh sau trên Terminal:
```powershell
python main.py
# Hoặc: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Sau khi server báo sẵn sàng:
👉 Mở trình duyệt Web (Chrome, Edge, Firefox) và truy cập: **[http://localhost:8000/mockup](http://localhost:8000/mockup)**

---

## 🖥️ Trải Nghiệm Trên Giao Diện Mockup Studio

Giao diện `mockup.html` cung cấp đầy đủ công cụ test trực quan:
1. **Kéo & Thả ảnh siêu âm**: Hỗ trợ định dạng `.jpg`, `.png`, `.bmp`.
2. **Chọn chế độ**:
   * **✨ Gemini Caliper Remover (Chuyên Sâu)**: Kiểm tra trực tiếp thuật toán xóa Caliper bằng Gemini, so sánh 3 panel: *Ảnh Gốc*, *Sub-pixel Mask*, *Kết Quả Phục Hồi*.
   * **🔬 Full AI Pipeline**: Chạy toàn bộ chuỗi tiền xử lý Safe Area + Caliper Removal + Lọc nhiễu SRAD + Phân vùng MedSAM.
3. **Bảng đo lường chất lượng (Metrics Dashboard)**:
   * *Số Caliper phát hiện*: Đếm số lượng dấu caliper trên ảnh.
   * *Mức nhiễu hạt ($\sigma$)*: Hiển thị độ nhám hạt siêu âm được bảo toàn.
   * *Chuẩn màu*: Xác nhận trạng thái đơn sắc chống ngả màu RGB.
   * *Sai số vùng ngoài (MSE)*: Cam kết $0.00$ (không chỉnh sửa bất cứ gì ở vùng ngoài).

---

## 🧪 Chạy Bộ Kiểm Thử Tự Động (Automated Unit Tests)

Để kiểm chứng tự động các ràng buộc kỹ thuật (Background Invariance, Noise Preservation, Monochrome Consistency):
```powershell
.\venv\Scripts\python.exe -m unittest tests/test_gemini_caliper_remover.py
```
Kết quả mong đợi: **`Ran 4 tests in 0.12s ... OK`**

---

## 📡 API Endpoints Tham Khảo

### 1. Endpoint Xóa Caliper Chuyên Biệt (Gemini)
* **URL**: `POST /api/v1/ai/remove-calipers-gemini`
* **Content-Type**: `multipart/form-data`
* **Params**:
  * `file`: File ảnh siêu âm
  * `api_key` *(optional)*: Gemini API Key nếu không cấu hình trong `.env`
* **Response chuẩn JSON Wrapper**:
```json
{
  "success": true,
  "message": "Xóa caliper bằng LLM Gemini thành công.",
  "data": {
    "original_image_base64": "data:image/jpeg;base64,...",
    "processed_image_base64": "data:image/jpeg;base64,...",
    "caliper_mask_base64": "data:image/png;base64,...",
    "detected_boxes": [
      {"name": "caliper_cross", "xmin": 120, "ymin": 150, "xmax": 145, "ymax": 175}
    ],
    "metrics": {
      "status": "SUCCESS",
      "calipers_count": 1,
      "noise_std_estimated": 8.45,
      "is_monochrome_enforced": true,
      "background_difference_mse": 0.0
    }
  },
  "errorCode": null
}
```

### 2. cURL Example
```bash
curl -X POST "http://localhost:8000/api/v1/ai/remove-calipers-gemini" \
     -F "file=@sample_ultrasound.jpg" \
     -F "api_key=YOUR_GEMINI_API_KEY"
```

---

## 📂 Cấu Trúc Mã Nguồn

```text
Med_superApp_Python/
├── main.py                             # FastAPI server & API endpoints
├── mockup.html                         # Giao diện HTML Studio test trực quan
├── requirements.txt                    # Danh sách thư viện Python
├── .env                                # Chứa GEMINI_API_KEY (tùy chọn)
├── core_pipeline/
│   ├── gemini_caliper_remover.py       # Module lõi Gemini AI + Inpainting + Noise Synthesis
│   ├── preprocess.py                   # Tiền xử lý (Safe Area, OCR, Caliper)
│   ├── enhance.py                      # Bộ lọc khử nhiễu SRAD & chỉnh độ tương phản
│   ├── pipeline.py                     # Pipeline xử lý đơn & batch zip dataset
│   ├── augment.py                      # Module Data Augmentation
│   ├── xml_exporter.py                 # Trích xuất nhãn Pascal VOC XML
│   └── templates/                      # Mẫu template caliper (+, x) cho fallback
└── tests/
    └── test_gemini_caliper_remover.py  # Unit tests kiểm thử 4 ràng buộc kỹ thuật
```
