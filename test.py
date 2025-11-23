import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import VGG19_Weights
from PIL import Image, ImageFilter
import copy
import os

# === Thiết bị ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 1. Kích thước ảnh (CẬP NHẬT: 512 để nét hơn) ===
imsize = 512  

# === Kiểm tra định dạng ảnh ===
def check_image_format(image_path):
    valid_exts = ['.jpg', '.jpeg', '.png']
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in valid_exts:
        raise ValueError(f"❌ Định dạng ảnh không hợp lệ: {ext}. Chỉ hỗ trợ {valid_exts}")

# === Load ảnh ===
def image_loader(image_input, imsize, device):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        # Bỏ kênh Alpha (nếu có) để chỉ lấy RGB
        transforms.Lambda(lambda x: x[:3, :, :] if x.shape[0] > 3 else x),
    ])
    
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
    else:
        image = image_input.convert('RGB')
        
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# === Chuẩn hóa ===
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std

# === Content Loss ===
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

# === Style Loss ===
def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

# === 2. HÀM MỚI: Total Variation Loss (Làm mịn ảnh) ===
def total_variation_loss(img, weight):
    """
    Tính toán sự chênh lệch giữa các pixel liền kề để giảm nhiễu.
    """
    b, c, h, w = img.size()
    # Chênh lệch theo chiều ngang (w)
    tv_h = torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    # Chênh lệch theo chiều dọc (h)
    tv_w = torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    return weight * (tv_h + tv_w)

# === Load model VGG19 ===
cnn = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
cnn_normalization_mean = [0.485, 0.456, 0.406]
cnn_normalization_std = [0.229, 0.224, 0.225]

# === Các layer sử dụng ===
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# === Xây dựng mô hình tính loss ===
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f'content_loss_{i}', content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f'style_loss_{i}', style_loss)
            style_losses.append(style_loss)

    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], ContentLoss) or isinstance(model[j], StyleLoss):
            break
    model = model[:j + 1]

    return model, style_losses, content_losses

# === Chuyển phong cách (Core Logic) ===
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1e6, content_weight=1e0, 
                       tv_weight=1e-4): # <--- Thêm tham số tv_weight
    
    print("🔧 Bắt đầu chuyển phong cách...")
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img)
    input_img.requires_grad_(True)
    
    # Dùng LBFGS để tối ưu hóa
    optimizer = optim.LBFGS([input_img])

    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1) # Giữ giá trị pixel trong khoảng [0, 1]
            
            optimizer.zero_grad()
            model(input_img)
            
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            
            # 3. Tính Total Variation Loss
            tv_score = total_variation_loss(input_img, tv_weight)
            
            # Tổng hợp Loss
            loss = style_score * style_weight + content_score * content_weight + tv_score
            loss.backward()
            
            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}: Style Loss: {style_score.item():.4f} | "
                      f"Content: {content_score.item():.4f} | TV Loss: {tv_score.item():.4f}")
            return loss
            
        optimizer.step(closure)
        
    with torch.no_grad():
        input_img.clamp_(0, 1)
        
    return input_img

# === HÀM WRAPPER cho Web App ===
def style_transfer(content_image_input, style_image_input, num_steps=300, 
                   style_weight=1e6, content_weight=1e0, tv_weight=1e-4):
    """
    Hàm gọi từ app.py, thêm tham số tv_weight
    """
    # Load ảnh
    content_img = image_loader(content_image_input, imsize, device)
    style_img = image_loader(style_image_input, imsize, device)
    input_img = content_img.clone()

    # Chạy
    output_tensor = run_style_transfer(
        cnn, cnn_normalization_mean, cnn_normalization_std,
        content_img, style_img, input_img,
        num_steps=num_steps, 
        style_weight=style_weight, 
        content_weight=content_weight,
        tv_weight=tv_weight
    )

    # Convert sang PIL
    output_image = output_tensor.cpu().clone().squeeze(0)
    output_image = transforms.ToPILImage()(output_image)
    
    # Filter nhẹ thêm lần nữa (tùy chọn)
    output_image = output_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=2))
    
    return output_image