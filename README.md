# 🎨 Neural Style Transfer Web App

Ứng dụng web sử dụng Deep Learning để biến những bức ảnh bình thường của bạn thành tác phẩm nghệ thuật, dựa trên phong cách của các danh họa nổi tiếng (như Van Gogh, Picasso...).

Dự án được xây dựng bằng PyTorch (VGG19) và Streamlit.



## ✨ Tính năng chính

Chuyển đổi phong cách: Áp dụng phong cách nghệ thuật (Style) vào ảnh gốc (Content) trong khi giữ nguyên nội dung.
Giao diện trực quan: Web App dễ sử dụng, cho phép tải ảnh lên và chọn phong cách từ danh sách có sẵn.
Hỗ trợ GPU: Tự động nhận diện và sử dụng GPU (CUDA) để tăng tốc độ xử lý nếu có.
Tải xuống: Cho phép tải ảnh kết quả chất lượng cao về máy.

## 🛠️ Cài đặt

Để chạy dự án trên máy cá nhân (Localhost), hãy làm theo các bước sau:

### 1. Clone dự án
```bash
git clone https://github.com/Ducdeptraicogisai/neural_style_transfer.git
cd neural-style-transfer
```

### 2. Cài đặt môi trường
Khuyên dùng Python 3.10 trở lên.
```bash
# Tạo môi trường ảo (Khuyên dùng)
python -m venv venv

# Kích hoạt môi trường (Windows)
.\venv\Scripts\activate

# Kích hoạt môi trường (Mac/Linux)
source venv/bin/activate

```
### 3. Cài đặt thư viện
```bash
pip install -r requirements.txt
```
Lưu ý: Nếu bạn có GPU NVIDIA, hãy cài đặt PyTorch phiên bản hỗ trợ CUDA để chạy nhanh hơn.

🚀 Hướng dẫn sử dụng
Chạy lệnh sau để khởi động ứng dụng web:

```bash
streamlit run app.py
```

Trình duyệt sẽ tự động mở địa chỉ http://localhost:8501.

Bước 1: Chọn một Style mẫu từ thanh bên trái (Sidebar).

Bước 2: Tải ảnh của bạn lên (Content Image).

Bước 3: Nhấn nút "🚀 Chuyển đổi phong cách" và chờ AI xử lý.

Bước 4: Tải ảnh kết quả về.


📂 Cấu trúc dự án
```plaintext
neural-style-transfer/
├── app.py               # Giao diện web chính (Streamlit)
├── test.py              # Thư viện xử lý thuật toán (VGG19, Loss Functions)
├── requirements.txt     # Danh sách thư viện cần thiết
├── style/               # Thư mục chứa ảnh phong cách mẫu
│   ├── style.png
│   ├── style_1.jpg
│   └── ...
├── content/             # Thư mục chứa ảnh nội dung mẫu (tùy chọn)
└── README.md            # Tài liệu hướng dẫn
```

## 🧠 Công nghệ sử dụng

- Python — Ngôn ngữ lập trình chính
- PyTorch — Framework Deep Learning dùng để load mô hình VGG19 và tính toán Loss
- Streamlit — Framework xây dựng giao diện web nhanh chóng cho Data Science
- VGG19 (Pre-trained) — Mô hình CNN dùng để trích xuất đặc trưng ảnh

## 📝 Thêm Style mới 
Bạn muốn thêm phong cách mới vào ứng dụng? Rất đơn giản:
- Tìm một bức ảnh nghệ thuật bạn thích (.jpg, .png).
- Copy file ảnh đó vào thư mục style/.
- Khởi động lại ứng dụng (hoặc bấm Rerun), style mới sẽ tự động hiện trong danh sách chọn.