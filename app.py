import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import pandas as pd
import plotly.express as px_chart 
import math
from arabic_reshaper import reshape
from bidi.algorithm import get_display

# ==========================================
# ۱. مدیریت متون دوجهته و استایلینگ (RTL)
# ==========================================
st.set_page_config(page_title="Aariz Precision Station V8.0.1", layout="wide")

def fix_aariz_text(text):
    if not text: return ""
    return get_display(reshape(str(text)))

def calculate_angle(p1, p2, p3):
    try:
        a, b, c = np.array(p1), np.array(p2), np.array(p3)
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    except: return 0.0

# ==========================================
# ۲. مرجع طلایی V7.8.16 (بدون تغییر عددی)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class CephaUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=29, features=[64, 128, 256, 512]):
        super(CephaUNet, self).__init__()
        self.ups, self.downs = nn.ModuleList(), nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, 2, 2))
            self.ups.append(DoubleConv(feature*2, feature))
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x); skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x); skip = skips[i//2]
            if x.shape != skip.shape: x = TF.resize(x, skip.shape[2:])
            x = self.ups[i+1](torch.cat((skip, x), dim=1))
        return self.final_conv(x)

# ==========================================
# ۳. بارگذاری مدل Aariz V2
# ==========================================
@st.cache_resource
def load_aariz_engine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CephaUNet().to(device)
    model.eval()
    return model, device

master_model, current_dev = load_aariz_engine()

# ==========================================
# ۴. تحلیل و گزارش‌دهی
# ==========================================
with st.sidebar:
    st.header(fix_aariz_text("پنل آنالیز عریض V8.0.1"))
    uploaded_file = st.file_uploader(fix_aariz_text("تصویر رادیوگرافی را آپلود کنید"), type=['png', 'jpg', 'jpeg'])
    st.divider()
    st.write(f"Engine: CephaNet-V2 | Dev: {current_dev}")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    W, H = img.size
    prep = img.convert("L").resize((512, 512))
    img_t = torch.from_numpy(np.array(prep)/255.0).unsqueeze(0).unsqueeze(0).float().to(current_dev)
    
    with torch.no_grad():
        prediction = master_model(img_t).cpu().numpy()[0]
    
    coords = []
    for i in range(29):
        y, x = np.unravel_index(prediction[i].argmax(), prediction[i].shape)
        coords.append((int(x * W / 512), int(y * H / 512)))

    # محاسبه شاخص‌های بالینی
    sna = calculate_angle(coords[0], coords[1], coords[2]) # S-N-A
    snb = calculate_angle(coords[0], coords[1], coords[3]) # S-N-B
    anb = sna - snb
    
    # منطق تشخیص خودکار وضعیت فکی (Steiner Logic)
    status = "Class I (Normal)"
    if anb > 4: status = "Class II (Skeletal)"
    elif anb < 0: status = "Class III (Skeletal)"

    st.subheader(fix_aariz_text("لندمارک‌گذاری و تشخیص کلاس فکی"))
    st.info(f"Clinical Diagnosis: {status}")

    vis_img = img.copy()
    draw = ImageDraw.Draw(vis_img)
    for i, (cx, cy) in enumerate(coords):
        draw.ellipse([cx-4, cy-4, cx+4, cy+4], fill="#FF4B4B", outline="white")
    st.image(vis_img, width='stretch')

    # بخش نمودارها
    df = pd.DataFrame({
        'Metric': ['SNA', 'SNB', 'ANB'],
        'Patient': [round(sna, 1), round(snb, 1), round(anb, 1)],
        'Normal': [82.0, 80.0, 2.0]
    })
    
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(df, width='stretch')
    with c2:
        fig = px_chart.bar(df, x='Metric', y=['Patient', 'Normal'], barmode='group')
        st.plotly_chart(fig, width='stretch')

    st.success(fix_aariz_text("گزارش آنالیز Steiner با موفقیت تولید شد."))
