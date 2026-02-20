import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import os
import urllib.request
from PIL import Image, ImageDraw

# --- ۱. تعریف معماری مدل UNet ---
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

# --- ۲. مدیریت دانلود و بارگذاری مدل‌ها (Ensemble) ---
@st.cache_resource
def load_all_models():
    device = torch.device('cpu')
    os.makedirs('models', exist_ok=True)
    
    # لینک‌های مستقیم مدل‌ها (باید لینک مستقیم دانلود را اینجا جایگزین کنید)
    # اگر مدل‌ها در درایو هستند، باید لینک Direct Download بسازید
    model_urls = {
        'general': 'لینک_مستقیم_مدل_عمومی',
        'specialist': 'لینک_مستقیم_مدل_اسپشیالیست',
        'tmj': 'لینک_مستقیم_مدل_tmj'
    }
    
    checkpoints = {
        'general': 'models/checkpoint_unet_clinical.pth',
        'specialist': 'models/specialist_pure_model.pth',
        'tmj': 'models/tmj_specialist_model.pth'
    }
    
    models = {}
    for name, path in checkpoints.items():
        # اگر فایل در سرور استریم‌لیت نبود، دانلود شود
        if not os.path.exists(path):
            with st.spinner(f'در حال دانلود مدل {name}... (این کار فقط یکبار انجام می‌شود)'):
                # urllib.request.urlretrieve(model_urls[name], path) # غیرفعال تا زمان جایگزینی لینک
                pass 
        
        model = UNet(n_channels=1, n_classes=29)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models[name] = model
    return models, device

# --- ۳. توابع محاسباتی آنالیز Steiner و McNamara ---
def get_landmarks(heatmaps):
    landmark_names = [
        'Sella', 'Nasion', 'A-point', 'B-point', 'Pogonion', 'Menton', 'Gnathion', 
        'Gonion', 'Orbitale', 'Porion', 'Condylion', 'Articulare', 'ANS', 'PNS',
        'Upper Incisor Tip', 'Lower Incisor Tip', 'Soft Tissue Nasion', 'Tip of Nose', 
        'Soft Tissue Menton', 'TMJ_Point', 'Ricketts_Point' # و مابقی تا ۲۹ نقطه
    ]
    landmarks = {}
    for i in range(min(len(landmark_names), heatmaps.shape[1])):
        heatmap = heatmaps[0, i].cpu().numpy()
        _, _, _, max_loc = cv2.minMaxLoc(heatmap)
        landmarks[landmark_names[i]] = max_loc
    return landmarks

def calculate_ortho_analysis(pts, pixel_size):
    def get_angle(p1, p2, p3):
        v1, v2 = np.array(p1) - np.array(p2), np.array(p3) - np.array(p2)
        dot = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

    res = {}
    try:
        if all(k in pts for k in ['Sella', 'Nasion', 'A-point']):
            res['SNA'] = get_angle(pts['Sella'], pts['Nasion'], pts['A-point'])
        if all(k in pts for k in ['Sella', 'Nasion', 'B-point']):
            res['SNB'] = get_angle(pts['Sella'], pts['Nasion'], pts['B-point'])
        if 'SNA' in res and 'SNB' in res:
            res['ANB'] = res['SNA'] - res['SNB']
        if all(k in pts for k in ['Condylion', 'A-point']):
            res['McNamara_Length'] = np.linalg.norm(np.array(pts['Condylion']) - np.array(pts['A-point'])) * pixel_size
    except: pass
    return res

# --- ۴. بدنه اصلی برنامه Streamlit ---
st.set_page_config(page_title="CephRad AI", layout="centered")
st.title("🦷 CephRad: آنالیز هوشمند رادیوگرافی")

uploaded_file = st.file_uploader("تصویر سفالومتری را انتخاب کنید", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="تصویر ورودی", use_column_width=True)

    if st.button("🚀 اجرای آنالیز کلینیکی"):
        with st.spinner("پردازش توسط مدل‌های Ensemble..."):
            models, device = load_all_models()
            gray_img = img.convert('L').resize((512, 512))
            input_tensor = torch.from_numpy(np.array(gray_img)).float().unsqueeze(0).unsqueeze(0) / 255.0
            
            with torch.no_grad():
                # ترکیب خروجی سه مدل
                out = (models['general'](input_tensor) + models['specialist'](input_tensor) + models['tmj'](input_tensor)) / 3.0
            
            # خواندن ضریب تبدیل
            pixel_size = 0.1 # مقدار پیش‌فرض
            if os.path.exists('mappings.csv'):
                df = pd.read_csv('mappings.csv')
                match = df[df['image_name'] == uploaded_file.name]
                if not match.empty: pixel_size = match['pixel_size'].values[0]

            pts = get_landmarks(out)
            analysis = calculate_ortho_analysis(pts, pixel_size)
            
            # رسم لندمارک‌ها
            
            draw = ImageDraw.Draw(img)
            for name, p in pts.items():
                draw.ellipse((p[0]-5, p[1]-5, p[0]+5, p[1]+5), fill='red', outline='white')
            st.image(img, caption="لندمارک‌های استخراج شده", use_column_width=True)

            # نمایش نتایج در کارت‌های رنگی
            st.subheader("📋 گزارش نهایی")
            c1, c2, c3 = st.columns(3)
            c1.metric("SNA", f"{analysis.get('SNA', 0):.1f}°")
            c2.metric("SNB", f"{analysis.get('SNB', 0):.1f}°")
            c3.metric("ANB", f"{analysis.get('ANB', 0):.1f}°")
            st.info(f"📏 طول موثر فک بالا (McNamara): {analysis.get('McNamara_Length', 0):.2f} mm")
