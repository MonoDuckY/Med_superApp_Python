# THÔNG TIN DỰ ÁN (Dùng chung cho cả 3 Repos)
- **Dự án:** Ứng dụng chẩn đoán hình ảnh y tế.
- **Nền tảng Client:** Web & Mobile.
- **Quy mô nhóm:** 5 thành viên.
- **Quy tắc cơ sở dữ liệu:** KHÔNG lưu trực tiếp file ảnh vào Database (chỉ lưu URL/Metadata). Giới hạn Document 16MB.

---

# QUY CHUẨN GIAO TIẾP API (BẮT BUỘC)
Mọi API trả về từ Spring Boot và nhận tại Flutter/Python đều phải tuân thủ nghiêm ngặt định dạng JSON bọc (Wrapper) sau:
{
  "success": boolean,
  "message": string,
  "data": object | array | null,
  "errorCode": string | null
}

---

# QUY CHUẨN KỸ THUẬT RIÊNG (Tùy biến theo từng Repo)

## Nếu đây là Frontend Repo (Flutter):
- **Kiến trúc:** MVVM (Model - View - ViewModel). Tách biệt hoàn toàn giao diện và logic.
- **State Management:** Sử dụng `Provider`.
- **Network Client:** Sử dụng `dio` thay cho `http`.
- **Cấu trúc thư mục:** `lib/core`, `lib/models`, `lib/services`, `lib/viewmodels`, `lib/views`.

## Nếu đây là Backend Repo (Java Spring Boot):
- **Công cụ Build:** Gradle (Groovy DSL).
- **Kiến trúc:** 3-Tier Architecture (Controllers - Services - Repositories).
- **Ngôn ngữ:** Java 17 hoặc 21. 
- **Cấu trúc thư mục:** Nhóm theo tính năng (Package by Feature) hoặc theo lớp (Package by Layer).
- **Lưu ý:** Bắt buộc sử dụng Generic `ApiResponse<T>` cho mọi Controller.

## Nếu đây là AI Service Repo (Python):
- **Framework:** FastAPI.
- **Môi trường:** Bắt buộc sử dụng môi trường ảo (`venv`). Không push thư mục `venv` lên Git.
- **Xử lý Request:** Nhận file ảnh qua `UploadFile`, trả kết quả nhanh chóng, không thực hiện các tác vụ block luồng chính.
- **Lưu ý trọng số AI:** Các file `.h5`, `.pt`, `.onnx` phải được khai báo trong `.gitignore`.