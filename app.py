import streamlit as st
import os
import io
from PIL import Image
import torch
from torchvision import models, transforms
from test import style_transfer
# Import các hàm từ test.py của bạn (bạn cần chỉnh sửa test.py một chút để import được)
# Giả sử bạn đã refactor test.py thành module hoặc copy code cần thiết sang đây.
from test import run_style_transfer, image_loader, cnn_normalization_mean, cnn_normalization_std

# --- CẤU HÌNH ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMSIZE = 512
STYLE_FOLDER = 'style'

# --- CACHE MODEL ---
# Dùng cache để không phải load lại VGG19 mỗi lần người dùng bấm nút
@st.cache_resource
def load_vgg_model():
    cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(DEVICE).eval()
    return cnn

cnn = load_vgg_model()

# --- GIAO DIỆN WEB ---
st.title("🎨 AI Art - Neural Style Transfer")
st.write("Biến bức ảnh của bạn thành tác phẩm nghệ thuật!")

# 1. Cột bên trái: Chọn Style
with st.sidebar:
    st.header("1. Chọn phong cách")
    # Lấy danh sách file trong thư mục style
    try:
        style_files = [f for f in os.listdir(STYLE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        selected_style_name = st.selectbox("Chọn style mẫu:", style_files)
        
        if selected_style_name:
            style_path = os.path.join(STYLE_FOLDER, selected_style_name)
            st.image(style_path, caption="Style đã chọn", width="stretch")
    except FileNotFoundError:
        st.error(f"Không tìm thấy thư mục '{STYLE_FOLDER}'")

# 2. Cột chính: Upload Content và Kết quả
st.header("2. Tải ảnh của bạn lên")
uploaded_file = st.file_uploader("Chọn ảnh nội dung (Content Image)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    content_image = Image.open(uploaded_file)
    st.image(content_image, caption="Ảnh gốc", width=400)

    # Nút bấm xử lý
    if st.button("🚀 Chuyển đổi phong cách"):
        with st.spinner('Đang vẽ... (Sẽ mất chút thời gian)'):
            try:
                # 1. Chạy thuật toán
                result_image = style_transfer(
                    content_image_input=content_image,
                    style_image_input=style_path,
                    num_steps=300 
                )
                
                # 2. Hiển thị kết quả
                st.success("Hoàn tất!")
                st.image(result_image, caption="Kết quả", width="stretch")
                
                # === PHẦN THÊM MỚI: Nút Download ===
                # Chuyển ảnh PIL thành bytes để tải về
                buf = io.BytesIO()
                result_image.save(buf, format="PNG")
                byte_im = buf.getvalue()

                st.download_button(
                    label="⬇️ Tải ảnh về máy",
                    data=byte_im,
                    file_name="neural_style_art.png",
                    mime="image/png"
                )
                # ===================================
                
            except Exception as e:
                st.error(f"Có lỗi xảy ra: {e}")