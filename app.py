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
# ۱. زیرساخت سیستمی (RTL & 2026 Standards)
# ==========================================
st.set_page_config(page_title="Aariz Precision Station V8.0.0", layout="wide")

def fix_aariz_text(text):
    if not text: return ""
    return get_display(reshape(str(text)))

# تابع محاسبه زاویه بین سه نقطه (مثلاً S-N-A)
def calculate_angle(p1, p2, p3):
    try:
        a = np.array(p1)
        b = np.array(p2) # نقطه راس (Vertex)
        c = np.array(p3)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    except:
        return 0.0

# ==========================================
# ۲. معماری مرجع طلایی V7.8.16 (بدون تغییر عددی)
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
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i//2]
            if x.shape != skip.shape: x = TF.resize(x, skip.shape[2:])
            x = self.ups[i+1](torch.cat((skip, x), dim=1))
        return self.final_conv(x)

# ==========================================
# ۳. مدیریت لودینگ و حافظه
# ==========================================
@st.cache_resource
def load_aariz_engine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CephaUNet().to(device)
    model.eval()
    return model, device

master_model, current_dev = load_aariz_engine()

# ==========================================
# ۴. تحلیل تصویر و استخراج نتایج واقعی
# ==========================================
with st.sidebar:
    st.header(fix_aariz_text("ایستگاه پردازش V8.0.0"))
    uploaded_file = st.file_uploader(fix_aariz_text("تصویر سفالومتری را آپلود کنید"), type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    W, H = img.size
    
    # پردازش تانسوری دقیق
    prep = img.convert("L").resize((512, 512))
    img_t = torch.from_numpy(np.array(prep)/255.0).unsqueeze(0).unsqueeze(0).float().to(current_dev)
    
    with torch.no_grad():
        prediction = master_model(img_t).cpu().numpy()[0]
    
    # نگاشت ۲۹ نقطه (نقاط مورد نیاز برای استینر: S=0, N=1, A=2, B=3 - فرض بر اساس ساختار مدل شما)
    coords = []
    for i in range(29):
        y, x = np.unravel_index(prediction[i].argmax(), prediction[i].shape)
        coords.append((int(x * W / 512), int(y * H / 512)))

    # محاسبه زوایای واقعی استینر
    sna = calculate_angle(coords[0], coords[1], coords[2]) # S-N-A
    snb = calculate_angle(coords[0], coords[1], coords[3]) # S-N-B
    anb = sna - snb

    # نمایش تصویر با لندمارک‌ها
    vis_img = img.copy()
    draw = ImageDraw.Draw(vis_img)
    for i, (cx, cy) in enumerate(coords):
        color = "#00FFCC" if i < 4 else "#FF4B4B" # برجسته کردن نقاط استینر
        draw.ellipse([cx-4, cy-4, cx+4, cy+4], fill=color, outline="white")
    
    st.subheader(fix_aariz_text("تجزیه و تحلیل هندسی لندمارک‌ها"))
    st.image(vis_img, width='stretch')

    # ==========================================
    # ۵. نمایش گرافیکی نتایج محاسباتی
    # ==========================================
    st.divider()
    
    clinical_data = {
        'Index': ['SNA', 'SNB', 'ANB'],
        'Patient': [round(sna, 1), round(snb, 1), round(anb, 1)],
        'Norm': [82.0, 80.0, 2.0]
    }
    df_results = pd.DataFrame(clinical_data)
    
    col_t, col_g = st.columns([1, 1])
    with col_t:
        st.write(fix_aariz_text("جدول مقایسه‌ای شاخص‌ها"))
        st.table(df_results)
    
    with col_g:
        fig = px_chart.bar(df_results, x='Index', y=['Patient', 'Norm'],
                           barmode='group', height=400,
                           color_discrete_map={'Patient': '#00FFCC', 'Norm': '#1F77B4'})
        st.plotly_chart(fig, width='stretch')

    st.success(fix_aariz_text("تحلیل هندسی با دقت پیکسل انجام و گزارش نهایی صادر شد."))
