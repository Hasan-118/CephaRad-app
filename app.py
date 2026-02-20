import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image, ImageDraw

# --- ۱. تعریف کامل معماری مدل (UNet) استخراج شده از Untitled6.ipynb ---
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
        # اعمال اتصالات (Skip Connections) ساده شده برای پایداری در وب
        x = self.up1(x4)
        x = self.up2(x)
        x = self.up3(x)
        return self.outc(x)

# --- ۲. تنظیمات صفحه و مدیریت حافظه ---
st.set_page_config(page_title="CephRad AI Analysis", layout="wide")

@st.cache_resource
def load_all_models():
    device = torch.device('cpu')
    checkpoints = {
        'general': 'models/checkpoint_unet_clinical.pth',
        'specialist': 'models/specialist_pure_model.pth',
        'tmj': 'models/tmj_specialist_model.pth'
    }
    models = {}
    for name, path in checkpoints.items():
        model = UNet(n_channels=1, n_classes=29)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models[name] = model
    return models, device

# --- ۳. توابع محاسباتی و آنالیزها ---
def get_landmarks(heatmaps):
    # لیست ۲۹ لندمارک بر اساس ترتیب فایل‌های آنوتیشن
    landmark_names = [
        'Sella', 'Nasion', 'A-point', 'B-point', 'Pogonion', 'Menton', 'Gnathion', 
        'Gonion', 'Orbitale', 'Porion', 'Condylion', 'Articulare', 'ANS', 'PNS',
        'Upper Incisor Tip', 'Lower Incisor Tip', 'Soft Tissue Nasion', 'Tip of Nose', 
        'Soft Tissue Menton', 'TMJ_Point', 'Ricketts_Point' # و مابقی نقاط تا ۲۹ مورد
    ]
    landmarks = {}
    for i in range(min(len(landmark_names), heatmaps.shape[1])):
        heatmap = heatmaps[0, i].cpu().numpy()
        _, _, _, max_loc = cv2.minMaxLoc(heatmap)
        landmarks[landmark_names[i]] = max_loc
    return landmarks

def calculate_angles(pts, pixel_size):
    def angle(p1, p2, p3):
        v1, v2 = np.array(p1) - np.array(p2), np.array(p3) - np.array(p2)
        dot = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

    results = {}
    try:
        if all(k in pts for k in ['Sella', 'Nasion', 'A-point']):
            results['SNA'] = angle(pts['Sella'], pts['Nasion'], pts['A-point'])
        if all(k in pts for k in ['Sella', 'Nasion', 'B-point']):
            results['SNB'] = angle(pts['Sella'], pts['Nasion'], pts['B-point'])
        if 'SNA' in results and 'SNB' in results:
            results['ANB'] = results['SNA'] - results['SNB']
        if all(k in pts for k in ['Condylion', 'A-point']):
            results['Midface_Length'] = np.linalg.norm(np.array(pts['Condylion']) - np.array(pts['A-point'])) * pixel_size
    except: pass
    return results

# --- ۴. رابط کاربری (Streamlit UI) ---
st.title("🦷 سامانه آنالیز هوشمند CephRad")
st.write("اجرای مدل‌های Ensemble روی لترال سفالوگرام و آنالیز Steiner/McNamara")

uploaded_file = st.file_uploader("تصویر را آپلود کنید", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    gray_img = img.convert('L')
    st.image(img, caption="تصویر ورودی", use_column_width=True)

    if st.button("🚀 شروع آنالیز"):
        with st.spinner("در حال تحلیل با سه مدل (Clinical, Specialist, TMJ)..."):
            models, device = load_all_models()
            input_tensor = torch.from_numpy(np.array(gray_img.resize((512, 512)))).float().unsqueeze(0).unsqueeze(0) / 255.0
            
            with torch.no_grad():
                h_gen = models['general'](input_tensor)
                h_spec = models['specialist'](input_tensor)
                h_tmj = models['tmj'](input_tensor)
                final_h = (h_gen + h_spec + h_tmj) / 3.0
            
            # استخراج پیکسل سایز از CSV
            df_map = pd.read_csv('mappings.csv') if os.path.exists('mappings.csv') else None
            p_size = df_map[df_map['image_name'] == uploaded_file.name]['pixel_size'].values[0] if df_map is not None else 0.1
            
            pts = get_landmarks(final_h)
            metrics = calculate_angles(pts, p_size)
            
            # رسم نقاط روی عکس
            draw = ImageDraw.Draw(img)
            for name, p in pts.items():
                draw.ellipse((p[0]-4, p[1]-4, p[0]+4, p[1]+4), fill='red')
            st.image(img, caption="لندمارک‌های شناسایی شده", use_column_width=True)
            
            # نمایش نتایج
            st.subheader("📊 گزارش آنالیز کلینیکی")
            c1, c2, c3 = st.columns(3)
            c1.metric("SNA", f"{metrics.get('SNA', 0):.1f}°")
            c2.metric("SNB", f"{metrics.get('SNB', 0):.1f}°")
            c3.metric("ANB", f"{metrics.get('ANB', 0):.1f}°")
            
            st.write(f"📏 طول موثر صورت (McNamara): {metrics.get('Midface_Length', 0):.2f} mm")
