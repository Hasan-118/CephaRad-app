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

# --- ۱. معماری دقیق UNet مطابق با فایل Untitled6.ipynb ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
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
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
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

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=29, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

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
        self.outc = OutConv(64, n_classes)

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
        logits = self.outc(x)
        return logits

# --- ۲. بارگذاری Ensemble از Google Drive ---
@st.cache_resource
def load_all_models():
    device = torch.device('cpu')
    os.makedirs('models', exist_ok=True)
    
    # آیدی‌های خود را جایگزین کنید (از حافظه برای دانلود استفاده می‌شود)
    drive_ids = {'general': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks', 'specialist': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU', 'tmj': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'}
    paths = {'general': 'models/mod1.pth', 'specialist': 'models/mod2.pth', 'tmj': 'models/mod3.pth'}
    
    models = {}
    for name, f_id in drive_ids.items():
        if not os.path.exists(paths[name]):
            gdown.download(f'https://drive.google.com/uc?id={f_id}', paths[name], quiet=False)
        model = UNet(n_channels=1, n_classes=29)
        model.load_state_dict(torch.load(paths[name], map_location=device))
        model.eval()
        models[name] = model
    return models, device

# --- ۳. پردازش تصویر و آنالیزهای کلینیکی ---
def process_data(final_output, original_size, pixel_size):
    names = ['Sella', 'Nasion', 'A-point', 'B-point', 'Pogonion', 'Menton', 'Gnathion', 
             'Gonion', 'Orbitale', 'Porion', 'Condylion', 'Articulare', 'ANS', 'PNS',
             'U1_Tip', 'L1_Tip', 'ST_Nasion', 'Nose_Tip', 'ST_Menton'] # تا ۲۹ مورد
    
    w_orig, h_orig = original_size
    scale_x, scale_y = w_orig / 512, h_orig / 512
    
    pts = {}
    for i in range(min(len(names), final_output.shape[1])):
        heatmap = final_output[0, i].detach().numpy()
        _, _, _, max_loc = cv2.minMaxLoc(heatmap)
        pts[names[i]] = (int(max_loc[0] * scale_x), int(max_loc[1] * scale_y))
    
    def get_angle(p1, p2, p3):
        v1, v2 = np.array(p1) - np.array(p2), np.array(p3) - np.array(p2)
        dot = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

    report = {}
    if all(k in pts for k in ['Sella', 'Nasion', 'A-point']): report['SNA'] = get_angle(pts['Sella'], pts['Nasion'], pts['A-point'])
    if all(k in pts for k in ['Sella', 'Nasion', 'B-point']): report['SNB'] = get_angle(pts['Sella'], pts['Nasion'], pts['B-point'])
    return pts, report

# --- ۴. اجرای Streamlit UI ---
st.set_page_config(page_title="CephRad AI", layout="centered")
st.title("🦷 آنالیز تخصصی CephRad")

file = st.file_uploader("آپلود تصویر سفالومتری", type=['png', 'jpg', 'jpeg'])

if file:
    img_orig = Image.open(file).convert('RGB')
    st.image(img_orig, use_column_width=True)

    if st.button("🚀 شروع آنالیز Ensemble"):
        models, device = load_all_models()
        tensor_in = torch.from_numpy(np.array(img_orig.convert('L').resize((512, 512)))).float().unsqueeze(0).unsqueeze(0) / 255.0
        
        with torch.no_grad():
            output = (models['general'](tensor_in) + models['specialist'](tensor_in) + models['tmj'](tensor_in)) / 3.0
            
        pts, analysis = process_data(output, img_orig.size, 0.1)
        
        draw = ImageDraw.Draw(img_orig)
        for p in pts.values(): draw.ellipse((p[0]-12, p[1]-12, p[0]+12, p[1]+12), fill='red', outline='white')
        st.image(img_orig, caption="نقطه‌گذاری نهایی اصلاح شده", use_column_width=True)
        
        c1, c2 = st.columns(2)
        c1.metric("SNA", f"{analysis.get('SNA', 0):.1f}°")
        c2.metric("SNB", f"{analysis.get('SNB', 0):.1f}°")

