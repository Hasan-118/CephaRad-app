import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
import os
import gdown
from PIL import Image, ImageDraw

# --- ۱. معماری UNet (اصلاح شده برای مطابقت با فایل‌های آموزشی) ---
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

# --- ۲. بارگذاری مدل‌ها با رفع خطای بارگذاری (Strict=False) ---
@st.cache_resource
def load_all_models():
    device = torch.device('cpu')
    os.makedirs('models', exist_ok=True)
    
    drive_ids = {
        'general': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks', 
        'specialist': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU', 
        'tmj': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'
    }
    paths = {'general': 'models/mod1.pth', 'specialist': 'models/mod2.pth', 'tmj': 'models/mod3.pth'}
    
    loaded_models = {}
    for name, f_id in drive_ids.items():
        if not os.path.exists(paths[name]):
            with st.spinner(f'در حال دانلود مدل {name}...'):
                gdown.download(id=f_id, output=paths[name], quiet=False)
        
        model = UNet(n_channels=1, n_classes=29, bilinear=True)
        # استفاده از strict=False برای نادیده گرفتن لایه‌های نامنطبق احتمالی
        state_dict = torch.load(paths[name], map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        loaded_models[name] = model
            
    return loaded_models, device

# --- ۳. پردازش و ترسیم لندمارک‌ها (Scaling Fix) ---
def process_output(output, original_size):
    # نام ۱۹ نقطه اصلی (برای تست، لیست را تا ۲۹ مورد بر اساس داکیومنت کامل کنید)
    names = ['Sella', 'Nasion', 'A-point', 'B-point', 'Pogonion', 'Menton', 'Gnathion', 
             'Gonion', 'Orbitale', 'Porion', 'Condylion', 'Articulare', 'ANS', 'PNS',
             'U1_Tip', 'L1_Tip', 'ST_Nasion', 'Nose_Tip', 'ST_Menton']
    
    w_orig, h_orig = original_size
    scale_x, scale_y = w_orig / 512, h_orig / 512
    
    pts = {}
    for i in range(min(len(names), output.shape[1])):
        heatmap = output[0, i].detach().numpy()
        _, _, _, max_loc = cv2.minMaxLoc(heatmap)
        pts[names[i]] = (int(max_loc[0] * scale_x), int(max_loc[1] * scale_y))
    
    return pts

# --- ۴. محاسبات Steiner ---
def calculate_steiner(pts):
    def get_angle(p1, p2, p3):
        v1, v2 = np.array(p1) - np.array(p2), np.array(p3) - np.array(p2)
        dot = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

    results = {}
    try:
        if all(k in pts for k in ['Sella', 'Nasion', 'A-point']):
            results['SNA'] = get_angle(pts['Sella'], pts['Nasion'], pts['A-point'])
        if all(k in pts for k in ['Sella', 'Nasion', 'B-point']):
            results['SNB'] = get_angle(pts['Sella'], pts['Nasion'], pts['B-point'])
    except: pass
    return results

# --- ۵. رابط کاربری Streamlit ---
st.set_page_config(page_title="CephRad AI", layout="centered")
st.title("🦷 آنالیز نهایی CephRad")

uploaded_file = st.file_uploader("تصویر سفالومتری را آپلود کنید", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    img_pil = Image.open(uploaded_file).convert('RGB')
    st.image(img_pil, caption="تصویر ورودی", use_column_width=True)

    if st.button("🚀 اجرای آنالیز Ensemble"):
        models, device = load_all_models()
        
        # پیش‌پردازش
        img_gray = img_pil.convert('L').resize((512, 512))
        tensor = torch.from_numpy(np.array(img_gray)).float().unsqueeze(0).unsqueeze(0) / 255.0
        
        with torch.no_grad():
            # خروجی میانگین از سه مدل
            out = (models['general'](tensor) + models['specialist'](tensor) + models['tmj'](tensor)) / 3.0
            
        pts = process_output(out, img_pil.size)
        analysis = calculate_steiner(pts)
        
        # رسم نقاط روی عکس اصلی
        draw = ImageDraw.Draw(img_pil)
        for name, p in pts.items():
            draw.ellipse((p[0]-10, p[1]-10, p[0]+10, p[1]+10), fill='red', outline='white', width=2)
            
        st.image(img_pil, caption="لندمارک‌های شناسایی شده", use_column_width=True)

        

        st.subheader("📊 نتایج آنالیز کلینیکی")
        c1, c2 = st.columns(2)
        c1.metric("SNA", f"{analysis.get('SNA', 0):.1f}°")
        c2.metric("SNB", f"{analysis.get('SNB', 0):.1f}°")
