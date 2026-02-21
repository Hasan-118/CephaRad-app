import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import gdown
from PIL import Image, ImageDraw

# --- ۱. معماری دقیق UNet (تطبیق لایه‌ها با مدل‌های آموزش دیده شما) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=29, bilinear=True):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# --- ۲. بارگذاری و ترکیب مدل‌ها (Ensemble Logic) ---
@st.cache_resource
def load_ensemble():
    device = torch.device('cpu')
    drive_ids = {
        'general': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks', 
        'specialist': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU', 
        'tmj': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'
    }
    os.makedirs('models', exist_ok=True)
    models = []
    for name, fid in drive_ids.items():
        path = f"models/{name}.pth"
        if not os.path.exists(path):
            gdown.download(id=fid, output=path, quiet=False)
        
        m = UNet(n_channels=1, n_classes=29)
        # بارگذاری وزن‌ها (مطمئن شوید bilinear در آموزش True بوده است)
        state_dict = torch.load(path, map_location=device)
        m.load_state_dict(state_dict, strict=False)
        m.eval()
        models.append(m)
    return models

# --- ۳. استخراج مختصات بر اساس ماکزیمم هیت‌مپ ---
def get_landmarks(models, image_tensor, original_size):
    with torch.no_grad():
        # Ensemble: جمع‌بندی خروجی مدل‌ها (Logits) و سپس میانگین‌گیری
        all_outputs = [m(image_tensor) for m in models]
        avg_output = torch.mean(torch.stack(all_outputs), dim=0)
        
        # اعمال Sigmoid برای نرمال‌سازی هیت‌مپ‌ها
        avg_output = torch.sigmoid(avg_output)
    
    w_orig, h_orig = original_size
    # مهم: مدل روی ۵۱۲ آموزش دیده، پس باید مختصات را به سایز اصلی مپ کنیم
    scale_x, scale_y = w_orig / 512, h_orig / 512
    
    landmarks = []
    output_np = avg_output[0].cpu().numpy() # [29, 512, 512]
    
    for i in range(29):
        heatmap = output_np[i]
        # پیدا کردن پیکسل با بیشترین مقدار (نقطه لندمارک)
        _, _, _, max_loc = cv2.minMaxLoc(heatmap)
        # تبدیل مختصات به سایز واقعی تصویر
        landmarks.append((int(max_loc[0] * scale_x), int(max_loc[1] * scale_y)))
    
    return landmarks

# --- ۴. رابط کاربری (Streamlit) ---
st.title("🦷 آنالیز دقیق CephRad Ensemble")

uploaded_file = st.file_uploader("آپلود تصویر سفالومتری", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    
    # پیش‌پردازش دقیق مطابق نوت‌بوک
    img_gray = img.convert('L').resize((512, 512))
    img_tensor = torch.from_numpy(np.array(img_gray)).float().unsqueeze(0).unsqueeze(0) / 255.0
    
    if st.button("🚀 اجرای انسامبل و نقطه‌گذاری"):
        models = load_ensemble()
        pts = get_landmarks(models, img_tensor, (w, h))
        
        # رسم نقاط روی تصویر اصلی
        draw = ImageDraw.Draw(img)
        for i, (px, py) in enumerate(pts):
            # دایره کوچک برای لندمارک
            draw.ellipse([px-10, py-10, px+10, py+10], fill='red', outline='white')
            # شماره‌گذاری لندمارک برای چک کردن ترتیب
            draw.text((px+12, py-12), str(i+1), fill='yellow')
            
        st.image(img, caption="لندمارک‌های شناسایی شده (Ensemble)", use_column_width=True)
