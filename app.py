import streamlit as st
import os
import io
from PIL import Image
import torch
from torchvision import models

# Import hàm xử lý chính từ file thư viện test.py
from test import style_transfer

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
st.set_page_config(page_title="AI Art Generator", page_icon="🎨") # Cấu hình tab trình duyệt
st.title("🎨 AI Art - Neural Style Transfer")
st.write("Biến bức ảnh của bạn thành tác phẩm nghệ thuật!")

# 1. Cột bên trái: Chọn Style & Tham số
with st.sidebar:
    st.header("1. Cấu hình")
    
    # --- Chọn ảnh Style ---
    st.subheader("Chọn phong cách mẫu")
    if not os.path.exists(STYLE_FOLDER):
        st.error(f"Không tìm thấy thư mục '{STYLE_FOLDER}'")
        selected_style_name = None
    else:
        style_files = [f for f in os.listdir(STYLE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        selected_style_name = st.selectbox("Danh sách style:", style_files)
        
        if selected_style_name:
            style_path = os.path.join(STYLE_FOLDER, selected_style_name)
            st.image(style_path, caption="Style đã chọn", width="stretch")

    st.markdown("---") # Đường kẻ ngang phân cách
    
    # --- Các thanh trượt tham số (Sliders) ---
    st.subheader("Tinh chỉnh tham số")
    
    # Slider 1: Độ mạnh của Style (Mặc định 1.000.000)
    style_weight = st.slider(
        "Độ mạnh Style (Style Weight)", 
        min_value=10000, 
        max_value=2000000, 
        value=1000000, 
        step=10000,
        help="Càng cao thì ảnh càng giống tranh vẽ, càng thấp thì càng giống ảnh gốc."
    )
    
    # Slider 2: Số bước lặp (Mặc định 300)
    num_steps = st.slider(
        "Số bước xử lý (Steps)", 
        min_value=50, 
        max_value=500, 
        value=300, 
        step=50,
        help="Số lần AI vẽ lại ảnh. Cao hơn = đẹp hơn nhưng lâu hơn."
    )
    
    # Slider 3: Độ mịn (Mặc định 0.0001)
    tv_weight = st.slider(
        "Độ mịn/Khử nhiễu (TV Weight)", 
        min_value=0.0, 
        max_value=0.001, 
        value=0.0001, 
        step=0.00001, 
        format="%.5f",
        help="Giảm nhiễu hạt và làm mượt các mảng màu."
    )

# 2. Cột chính: Upload Content và Kết quả
st.header("2. Tải ảnh của bạn lên")
uploaded_file = st.file_uploader("Chọn ảnh nội dung (Content Image)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Hiển thị ảnh gốc
    content_image = Image.open(uploaded_file)
    st.image(content_image, caption="Ảnh gốc", width=400)

    # Nút bấm xử lý
    if st.button("🚀 Chuyển đổi phong cách", type="primary"):
        if not selected_style_name:
             st.warning("Vui lòng chọn một Style ở cột bên trái trước!")
        else:
            with st.spinner(f'Đang vẽ với {num_steps} bước... (Sẽ mất chút thời gian)'):
                try:
                    # --- GỌI HÀM XỬ LÝ ---
                    # Truyền các tham số từ Slider vào hàm
                    result_image = style_transfer(
                        content_image_input=content_image,
                        style_image_input=style_path,
                        num_steps=num_steps,       # <--- Lấy từ slider
                        style_weight=style_weight, # <--- Lấy từ slider
                        tv_weight=tv_weight        # <--- Lấy từ slider
                    )
                    
                    # --- HIỂN THỊ KẾT QUẢ ---
                    st.success("Hoàn tất!")
                    st.image(result_image, caption="Kết quả nghệ thuật", width="stretch")
                    
                    # --- NÚT TẢI VỀ ---
                    buf = io.BytesIO()
                    result_image.save(buf, format="PNG")
                    byte_im = buf.getvalue()

                    st.download_button(
                        label="⬇️ Tải ảnh về máy",
                        data=byte_im,
                        file_name="neural_style_art.png",
                        mime="image/png"
                    )
                    
                except Exception as e:
                    st.error(f"Có lỗi xảy ra: {e}")
                    # In chi tiết lỗi ra console để debug nếu cần
                    print(e)