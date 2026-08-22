# Caliper Cleanroom

Công cụ local để phát hiện dấu caliper trên ảnh siêu âm bằng template matching, tạo mask tự động và dùng LaMa inpainting để loại bỏ dấu đánh dấu.

## Pipeline

```text
Ảnh siêu âm → Template matching → Mask → LaMa inpainting → Ảnh kết quả
```

Ảnh được xử lý local, không gửi lên Gemini hay server bên ngoài.

## Cài đặt tự động

Yêu cầu: Windows, Python 3.10, Git. NVIDIA GPU là tùy chọn; CPU vẫn chạy được nhưng chậm hơn.

Sau khi clone project, chạy:

```text
setup_project.bat
```

Script sẽ tạo `.venv`, cài dependency, tải model `big-lama` và kiểm tra CUDA.

Không commit `.venv/`, `big-lama/`, `input/` hoặc `output/`; các thư mục này đã có trong `.gitignore`.

## Chạy giao diện web

1. Chạy `start_backend.bat`.
2. Mở `index.html` bằng trình duyệt.
3. Tải ảnh lên.
4. Tô thủ công vùng caliper hoặc để backend tự phát hiện bằng templates.
5. Bấm **XỬ LÝ ẢNH**.

Backend chạy tại `http://127.0.0.1:8000`. Kiểm tra tại `http://127.0.0.1:8000/health`. Tắt bằng `stop_backend.bat`.

## Xử lý hàng loạt

Đặt ảnh vào thư mục `input/`, sau đó chạy:

```text
run_test.bat
```

Kết quả được lưu vào `output/`. Các định dạng hỗ trợ gồm JPG, JPEG, PNG, BMP, WEBP, TIF và TIFF.

## GPU

Kiểm tra PyTorch:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Nếu trả về `True` và tên NVIDIA GPU, backend và batch processor tự động dùng CUDA. Nếu không, chương trình chạy bằng CPU.

## Templates

Template phát hiện dấu nằm tại `highlight/templates/`. Có thể thêm hoặc thay template để cải thiện khả năng phát hiện. Ngưỡng matching hiện được cấu hình trong `process_images.py` và `server.py`.

## Lưu ý

LaMa tái tạo texture trong vùng mask, không khôi phục dữ liệu gốc tuyệt đối. Hãy kiểm tra kết quả thủ công; công cụ không thay thế đánh giá chuyên môn y khoa.
