import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import gdown
from PIL import Image, ImageDraw

# --- ۱. معماری استاندارد UNet (منطبق بر آموزش Untitled6) ---
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
        self.down4 = Down(512, 512)
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

# --- ۲. بارگذاری انسامبل از درایو ---
@st.cache_resource
def load_all_models():
    device = torch.device('cpu')
    os.makedirs('models', exist_ok=True)
    drive_ids = {
        'general': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks', 
        'specialist': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU', 
        'tmj': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'
    }
    paths = {k: f"models/{k}.pth" for k in drive_ids}
    models = {}
    for name, fid in drive_ids.items():
        if not os.path.exists(paths[name]):
            gdown.download(id=fid, output=paths[name], quiet=False)
        m = UNet(n_channels=1, n_classes=29)
        # استفاده از strict=False برای جلوگیری از خطای Runtime
        m.load_state_dict(torch.load(paths[name], map_location=device), strict=False)
        m.eval()
        models[name] = m
    return models

# --- ۳. پردازش و آنالیز Steiner ---
def process_and_analyze(pred, original_size):
    landmark_names = [
        'Sella', 'Nasion', 'A-point', 'B-point', 'Pogonion', 'Menton', 'Gnathion', 
        'Gonion', 'Orbitale', 'Porion', 'Condylion', 'Articulare', 'ANS', 'PNS',
        'U1_Tip', 'L1_Tip', 'ST_Nasion', 'Nose_Tip', 'ST_Menton'
    ]
    w, h = original_size
    sx, sy = w/512, h/512
    pts = {}
    
    for i in range(min(len(landmark_names), pred.shape[1])):
        heatmap = torch.sigmoid(pred[0, i]).detach().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7)
        heatmap[heatmap < 0.5] = 0
        _, _, _, max_loc = cv2.minMaxLoc(heatmap)
        pts[landmark_names[i]] = (int(max_loc[0]*sx), int(max_loc[1]*sy))
    
    def get_angle(p1, p2, p3):
        v1, v2 = np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)
        return np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6), -1, 1)))

    report = {}
    try:
        report['SNA'] = get_angle(pts['Sella'], pts['Nasion'], pts['A-point'])
        report['SNB'] = get_angle(pts['Sella'], pts['Nasion'], pts['B-point'])
        report['ANB'] = report['SNA'] - report['SNB']
    except: pass
    return pts, report

# --- ۴. رابط کاربری (UI) ---
st.set_page_config(page_title="CephRad AI", layout="centered")
st.title("🦷 پنل هوشمند CephRad")

# رفع مشکل رگولار اکسپرشن با استایل ساده
st.markdown("<style>.stMetric { background: #f0f2f6; padding: 10px; border-radius: 10px; }</style>", unsafe_allow_html=True)

file = st.file_uploader("تصویر سفالومتری (PNG, JPG)", type=['png', 'jpg', 'jpeg'])

if file:
    img_pil = Image.open(file).convert('RGB')
    st.image(img_pil, use_column_width=True)

    if st.button("🚀 شروع آنالیز Ensemble"):
        with st.spinner('در حال پردازش مدل‌های انسامبل...'):
            models = load_all_models()
            input_img = img_pil.convert('L').resize((512, 512))
            tensor = torch.from_numpy(np.array(input_img)).float().unsqueeze(0).unsqueeze(0) / 255.0
            
            with torch.no_grad():
                # میانگین‌گیری از هر ۳ مدل
                ensemble_pred = (models['general'](tensor) + models['specialist'](tensor) + models['tmj'](tensor)) / 3.0
            
            pts, results = process_and_analyze(ensemble_pred, img_pil.size)
            
            # رسم نقاط
            draw = ImageDraw.Draw(img_pil)
            for p in pts.values():
                draw.ellipse((p[0]-12, p[1]-12, p[0]+12, p[1]+12), fill='red', outline='white', width=3)
            
            st.image(img_pil, caption="نتیجه نهایی آنالیز", use_column_width=True)

            
            
            st.subheader("📊 نتایج کلینیکی")
            c1, c2, c3 = st.columns(3)
            c1.metric("SNA", f"{results.get('SNA', 0):.1f}°")
            c2.metric("SNB", f"{results.get('SNB', 0):.1f}°")
            c3.metric("ANB", f"{results.get('ANB', 0):.1f}°")
