import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import VGG19_Weights
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import copy
import os

# === Thiết bị ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Kích thước ảnh (tăng lên khi đã ổn định) ===
imsize = 512

# === Kiểm tra định dạng ảnh ===
def check_image_format(image_path):
    valid_exts = ['.jpg', '.jpeg', '.png']
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in valid_exts:
        raise ValueError(f"❌ Định dạng ảnh không hợp lệ: {ext}. Chỉ hỗ trợ {valid_exts}")

# === Load ảnh ===
def image_loader(image_path, imsize, device):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, :, :] if x.shape[0] > 3 else x),
    ])
    image = Image.open(image_path).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# === Hiển thị ảnh ===
def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    # Áp dụng bộ lọc làm nét mạnh hơn
    image = image.filter(ImageFilter.UnsharpMask(radius=3, percent=200, threshold=2))
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

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

    # Cắt model sau layer cuối cùng có loss
    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], ContentLoss) or isinstance(model[j], StyleLoss):
            break
    model = model[:j + 1]

    return model, style_losses, content_losses

# === Chuyển phong cách (dùng L-BFGS) ===
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1e6, content_weight=1e0):
    print("🔧 Bắt đầu chuyển phong cách...")
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img)
    input_img.requires_grad_(True)
    optimizer = optim.LBFGS([input_img])

    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_score * style_weight + content_score * content_weight
            loss.backward()
            if run[0] % 50 == 0:
                print(f"Step {run[0]}: Style Loss: {style_score.item():.4f} | Content Loss: {content_score.item():.4f}")
            run[0] += 1
            return loss
        optimizer.step(closure)
    with torch.no_grad():
        input_img.clamp_(0, 1)
    return input_img

# === ĐƯỜNG DẪN ẢNH ===
content_path = "content.png"  # Thay bằng đường dẫn ảnh của bạn
style_path = "style.png"      # Thay bằng đường dẫn ảnh của bạn

# === KIỂM TRA FILE ẢNH ===
check_image_format(content_path)
check_image_format(style_path)

# === LOAD ẢNH ===
content_img = image_loader(content_path, imsize, device)
style_img = image_loader(style_path, imsize, device)
input_img = content_img.clone()

# === CHẠY ===
output = run_style_transfer(
    cnn, cnn_normalization_mean, cnn_normalization_std,
    content_img, style_img, input_img,
    num_steps=50, style_weight=1e6, content_weight=1e0
)

# === HIỂN THỊ KẾT QUẢ ===
imshow(output, title="Ảnh đã chuyển phong cách")