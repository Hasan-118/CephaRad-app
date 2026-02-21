import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import os
import gdown  # برای دانلود مستقیم از گوگل درایو
from PIL import Image, ImageDraw

# --- ۱. معماری مدل UNet (استخراج شده از فایل Untitled6.ipynb) ---
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
        x = self.up1(x4)
        x = self.up2(x)
        x = self.up3(x)
        return self.outc(x)

# --- ۲. بارگذاری و دانلود مدل‌ها از گوگل درایو (Ensemble) ---
@st.cache_resource
def load_models_from_drive():
    device = torch.device('cpu')
    os.makedirs('models', exist_ok=True)
    
    # آیدی فایل‌ها در گوگل درایو (از لینک Share فایل‌ها استخراج کنید)
    drive_ids = {
        'general': 'آیدی_فایل_checkpoint_unet_clinical',
        'specialist': 'آیدی_فایل_specialist_pure_model',
        'tmj': 'آیدی_فایل_tmj_specialist_model'
    }
    
    paths = {
        'general': 'models/checkpoint_unet_clinical.pth',
        'specialist': 'models/specialist_pure_model.pth',
        'tmj': 'models/tmj_specialist_model.pth'
    }
    
    loaded_models = {}
    for name, file_id in drive_ids.items():
        if not os.path.exists(paths[name]):
            with st.spinner(f'در حال دانلود مدل {name} از گوگل درایو...'):
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, paths[name], quiet=False)
        
        model = UNet(n_channels=1, n_classes=29)
        model.load_state_dict(torch.load(paths[name], map_location=device))
        model.eval()
        loaded_models[name] = model
        
    return loaded_models, device

# --- ۳. استخراج مختصات و اصلاح Scaling (برای جلوگیری از تجمع نقاط) ---
def get_scaled_landmarks(output, original_size, input_size=(512, 512)):
    landmark_names = [
        'Sella', 'Nasion', 'A-point', 'B-point', 'Pogonion', 'Menton', 'Gnathion', 
        'Gonion', 'Orbitale', 'Porion', 'Condylion', 'Articulare', 'ANS', 'PNS',
        'U1_Tip', 'L1_Tip', 'Soft_Nasion', 'Nose_Tip', 'Soft_Menton' # لیست را تا ۲۹ کامل کنید
    ]
    
    w_orig, h_orig = original_size
    scale_x, scale_y = w_orig / input_size[0], h_orig / input_size[1]
    
    pts = {}
    for i in range(min(len(landmark_names), output.shape[1])):
        heatmap = output[0, i].detach().numpy()
        _, _, _, max_loc = cv2.minMaxLoc(heatmap)
        pts[landmark_names[i]] = (int(max_loc[0] * scale_x), int(max_loc[1] * scale_y))
    return pts

# --- ۴. آنالیز کلینیکی Steiner & McNamara ---
def compute_ortho_analysis(pts, pixel_size):
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
        if 'SNA' in results and 'SNB' in results:
            results['ANB'] = results['SNA'] - results['SNB']
    except: pass
    return results

# --- ۵. اجرای رابط کاربری Streamlit ---
st.set_page_config(page_title="CephRad AI Analysis", layout="wide")
st.title("🦷 آنالیز آنلاین CephRad")

uploaded_file = st.file_uploader("تصویر سفالومتری را انتخاب کنید", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    original_img = Image.open(uploaded_file).convert('RGB')
    st.image(original_img, caption="تصویر ورودی", use_column_width=True)

    if st.button("🚀 شروع آنالیز Ensemble"):
        models, device = load_models_from_drive()
        
        # پیش‌پردازش
        input_size = (512, 512)
        img_input = original_img.convert('L').resize(input_size)
        tensor_in = torch.from_numpy(np.array(img_input)).float().unsqueeze(0).unsqueeze(0) / 255.0
        
        with torch.no_grad():
            # ترکیب ۳ مدل (Ensemble)
            pred = (models['general'](tensor_in) + models['specialist'](tensor_in) + models['tmj'](tensor_in)) / 3.0
        
        # اصلاح مقیاس و نمایش نقاط
        pts = get_scaled_landmarks(pred, original_img.size)
        draw = ImageDraw.Draw(original_img)
        for name, p in pts.items():
            draw.ellipse((p[0]-8, p[1]-8, p[0]+8, p[1]+8), fill='red', outline='white')
        
        st.image(original_img, caption="لندمارک‌های شناسایی شده (اصلاح شده)", use_column_width=True)
        
        # نمایش آنالیز
        analysis = compute_ortho_analysis(pts, 0.1)
        st.subheader("📝 گزارش نهایی")
        c1, c2, c3 = st.columns(3)
        c1.metric("SNA", f"{analysis.get('SNA', 0):.1f}°")
        c2.metric("SNB", f"{analysis.get('SNB', 0):.1f}°")
        c3.metric("ANB", f"{analysis.get('ANB', 0):.1f}°")
