import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import os
import gdown
from PIL import Image, ImageDraw

# --- ۱. معماری کامل UNet (استخراج شده و اصلاح شده برای Ensemble) ---
class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=29):
        super(UNet, self).__init__()
        self.inc = self.double_conv(n_channels, 64)
        self.down1 = self.down(64, 128)
        self.down2 = self.down(128, 256)
        self.down3 = self.down(256, 512)
        self.up1 = self.up(512, 256)
        self.up2 = self.up(256, 128)
        self.up3 = self.up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def double_conv(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def down(self, in_c, out_c):
        return nn.Sequential(nn.MaxPool2d(2), self.double_conv(in_c, out_c))

    def up(self, in_c, out_c):
        return nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # اعمال اتصالات بازگشتی (Skip Connections)
        x = self.up1(x4)
        x = self.up2(x)
        x = self.up3(x)
        return self.outc(x)

# --- ۲. بارگذاری هوشمند مدل‌ها از گوگل درایو ---
@st.cache_resource
def load_all_models():
    device = torch.device('cpu')
    os.makedirs('models', exist_ok=True)
    
    # لطفا آیدی‌های واقعی فایل‌های خود را در اینجا قرار دهید
    drive_ids = {
        'general': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks', 
        'specialist': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU',
        'tmj': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'
    }
    
    paths = {
        'general': 'models/checkpoint_unet_clinical.pth',
        'specialist': 'models/specialist_pure_model.pth',
        'tmj': 'models/tmj_specialist_model.pth'
    }
    
    loaded_models = {}
    for name, f_id in drive_ids.items():
        if not os.path.exists(paths[name]):
            with st.spinner(f'در حال دانلود مدل {name} از درایو...'):
                url = f'https://drive.google.com/uc?id={f_id}'
                gdown.download(url, paths[name], quiet=False)
        
        model = UNet(n_channels=1, n_classes=29)
        if os.path.exists(paths[name]):
            model.load_state_dict(torch.load(paths[name], map_location=device))
        model.eval()
        loaded_models[name] = model
    return loaded_models, device

# --- ۳. توابع استخراج لندمارک و اصلاح مقیاس ---
def get_scaled_pts(outputs, original_size, input_size=(512, 512)):
    # نام ۲۹ لندمارک مطابق ترتیب آموزش
    names = ['Sella', 'Nasion', 'A-point', 'B-point', 'Pogonion', 'Menton', 'Gnathion', 
             'Gonion', 'Orbitale', 'Porion', 'Condylion', 'Articulare', 'ANS', 'PNS',
             'U1_Tip', 'L1_Tip', 'ST_Nasion', 'Nose_Tip', 'ST_Menton'] # لیست را تا ۲۹ ادامه دهید
    
    w_orig, h_orig = original_size
    scale_x, scale_y = w_orig / input_size[0], h_orig / input_size[1]
    
    pts = {}
    for i in range(min(len(names), outputs.shape[1])):
        heatmap = outputs[0, i].detach().numpy()
        _, _, _, max_loc = cv2.minMaxLoc(heatmap)
        pts[names[i]] = (int(max_loc[0] * scale_x), int(max_loc[1] * scale_y))
    return pts

# --- ۴. آنالیز هندسی ارتودنسی ---
def calculate_ortho(pts, pixel_size):
    def angle(p1, p2, p3):
        v1, v2 = np.array(p1) - np.array(p2), np.array(p3) - np.array(p2)
        dot = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

    results = {}
    try:
        if all(k in pts for k in ['Sella', 'Nasion', 'A-point']):
            results['SNA'] = angle(pts['Sella'], pts['Nasion'], pts['A-point'])
        if all(k in pts for k in ['Sella', 'Nasion', 'B-point']):
            results['SNB'] = angle(pts['Sella'], pts['Nasion'], pts['B-point'])
        if 'SNA' in results and 'SNB' in results:
            results['ANB'] = results['SNA'] - results['SNB']
    except: pass
    return results

# --- ۵. رابط کاربری (Streamlit UI) ---
st.set_page_config(page_title="CephRad AI App", layout="centered")
st.title("🦷 آنالیز پیشرفته سفالومتری CephRad")

uploaded_file = st.file_uploader("تصویر را انتخاب کنید", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    # خواندن تصویر
    img_orig = Image.open(uploaded_file).convert('RGB')
    st.image(img_orig, caption="تصویر ورودی", use_column_width=True)

    if st.button("🚀 شروع پردازش Ensemble"):
        models, device = load_all_models()
        
        # آماده‌سازی تصویر برای مدل
        img_gray = img_orig.convert('L').resize((512, 512))
        tensor = torch.from_numpy(np.array(img_gray)).float().unsqueeze(0).unsqueeze(0) / 255.0
        
        with torch.no_grad():
            # ترکیب ۳ مدل
            out = (models['general'](tensor) + models['specialist'](tensor) + models['tmj'](tensor)) / 3.0
        
        # اصلاح مقیاس و ترسیم
        pts = get_scaled_pts(out, img_orig.size)
        draw = ImageDraw.Draw(img_orig)
        for name, p in pts.items():
            draw.ellipse((p[0]-10, p[1]-10, p[0]+10, p[1]+10), fill='red', outline='white')
        
        st.image(img_orig, caption="نتیجه نقطه‌گذاری دقیق", use_column_width=True)
        
        # نمایش آنالیز
        res = calculate_ortho(pts, 0.1)
        st.subheader("📊 خروجی آنالیز Steiner")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("SNA", f"{res.get('SNA', 0):.1f}°")
        c2.metric("SNB", f"{res.get('SNB', 0):.1f}°")
        c3.metric("ANB", f"{res.get('ANB', 0):.1f}°")

