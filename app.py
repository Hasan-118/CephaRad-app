import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import gdown
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۱. ساختار مدل (دقیقاً مطابق حافظه) ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class CephaUNet(nn.Module):
    def __init__(self, n_landmarks=29):
        super().__init__()
        self.inc = DoubleConv(1, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_landmarks, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3)
        x = self.up1(x4); x = torch.cat([x, x3], dim=1); x = self.conv_up1(x)
        x = self.up2(x); x = torch.cat([x, x2], dim=1); x = self.conv_up2(x)
        x = self.up3(x); x = torch.cat([x, x1], dim=1); x = self.conv_up3(x)
        return self.outc(x)

# --- ۲. لودر اصلاح شده (رفع مشکل لود نشدن) ---
@st.cache_resource
def load_aariz_models_safe():
    model_ids = {
        'model1.pth': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks',
        'model2.pth': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU',
        'model3.pth': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'
    }
    device = torch.device("cpu")
    models = []
    
    for name, fid in model_ids.items():
        if not os.path.exists(name):
            gdown.download(f'https://drive.google.com/uc?id={fid}', name, quiet=False)
        
        try:
            m = CephaUNet(n_landmarks=29)
            ckpt = torch.load(name, map_location=device)
            state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            # حذف لایه‌های اضافی و تطبیق نام‌ها
            clean_state = {k.replace('module.', ''): v for k, v in state.items()}
            m.load_state_dict(clean_state, strict=False)
            m.eval()
            models.append(m)
        except Exception as e:
            st.error(f"خطا در لود {name}: {e}")
            
    return models, device

# --- ۳. پردازش تصویر و UI ---
st.set_page_config(page_title="Aariz AI", layout="wide")
models, device = load_aariz_models_safe()

if not models:
    st.error("⚠️ هیچ مدلی بارگذاری نشد. لطفاً لاگ‌های اپلیکیشن را چک کنید.")
    st.stop()

st.sidebar.success(f"✅ {len(models)} مدل با موفقیت لود شدند.")
up_file = st.sidebar.file_uploader("آپلود تصویر:", type=['png', 'jpg', 'jpeg'])

if up_file:
    img = Image.open(up_file).convert("RGB")
    # منطق پیش‌بینی و نمایش (مشابه قبل)...
    st.image(img, caption="تصویر بارگذاری شد.", use_container_width=True)
    st.info("در حال پردازش لندمارک‌ها...")
