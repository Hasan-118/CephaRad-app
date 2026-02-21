import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import gdown
from PIL import Image, ImageDraw
from collections import OrderedDict

# --- ۱. معماری UNet (دقیقاً مطابق پارامترهای نوت‌بوک CephaRad) ---
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
    def __init__(self, n_channels=1, n_classes=29):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512) # در برخی نسخه‌ها ۵۱۲ است، اگر خطا داد به ۱۰۲۴ تغییر دهید
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
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

# --- ۲. بارگذاری هوشمند (رفع مشکل لایه‌های Module) ---
def load_model_weights(model, path, device):
    state_dict = torch.load(path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k # حذف پیشوند module
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    return model

@st.cache_resource
def load_all_ensemble():
    device = torch.device('cpu')
    ids = {'gen': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks', 'spec': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU', 'tmj': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'}
    models = []
    for name, fid in ids.items():
        path = f"models/{name}.pth"
        os.makedirs('models', exist_ok=True)
        if not os.path.exists(path): gdown.download(id=fid, output=path, quiet=False)
        m = UNet(n_channels=1, n_classes=29)
        m = load_model_weights(m, path, device)
        m.eval()
        models.append(m)
    return models

# --- ۳. رابط کاربری و اجرای آنالیز ---
st.title("🦷 سامانه آنالیز هوشمند CephRad")

uploaded = st.file_uploader("تصویر را انتخاب کنید", type=['png', 'jpg', 'jpeg'])

if uploaded:
    img_orig = Image.open(uploaded).convert('RGB')
    w, h = img_orig.size
    
    # پیش‌پردازش (بسیار مهم: نرمال‌سازی منطبق بر دیتای Aariz)
    img_input = img_orig.convert('L').resize((512, 512))
    img_np = np.array(img_input).astype(np.float32) / 255.0
    # استانداردسازی (اگر در نوت‌بوک شما Mean/Std خاصی بود اینجا اعمال کنید)
    tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)

    if st.button("🚀 اجرای تحلیل انسامبل"):
        models = load_all_ensemble()
        with torch.no_grad():
            # میانگین‌گیری روی خروجی مدل‌ها (Ensemble)
            preds = [torch.sigmoid(m(tensor)) for m in models]
            final_pred = torch.mean(torch.stack(preds), dim=0).cpu().numpy()[0]

        draw = ImageDraw.Draw(img_orig)
        scale_x, scale_y = w / 512, h / 512
        
        for i in range(29):
            heatmap = final_pred[i]
            # پیدا کردن نقطه دقیق ماکزیمم
            _, max_val, _, max_loc = cv2.minMaxLoc(heatmap)
            
            # فقط اگر دقت نقطه از حدی بیشتر بود رسم شود (حذف نقاط پرت)
            if max_val > 0.1:
                cx, cy = int(max_loc[0] * scale_x), int(max_loc[1] * scale_y)
                draw.ellipse([cx-12, cy-12, cx+12, cy+12], fill='red', outline='white', width=3)

        st.image(img_orig, use_column_width=True)
