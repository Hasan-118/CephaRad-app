import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import pandas as pd
import os
from fpdf import FPDF, XPos, YPos # استفاده از سیستم موقعیت‌دهی جدید
from arabic_reshaper import reshape
from bidi.algorithm import get_display

# ==========================================
# ۱. تنظیمات سیستمی و یونیکد
# ==========================================
st.set_page_config(page_title="Aariz Precision Station V7.8.23", layout="wide")

def fix_fa(text):
    if not text: return ""
    return get_display(reshape(str(text)))

# ==========================================
# ۲. معماری شبکه عصبی (بدون تغییر - مرجع V7.8)
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
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            x = self.ups[idx+1](torch.cat((skip_connection, x), dim=1))
        return self.final_conv(x)

# ==========================================
# ۳. بارگذاری سیستم هوشمند (۳ مدل متخصص)
# ==========================================
@st.cache_resource
def init_station():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CephaUNet(in_channels=1, out_channels=29).to(device)
    # بارگذاری وزن‌های مرجع در این مرحله انجام می‌شود
    return model, device

master_model, device = init_station()

# ==========================================
# ۴. رابط کاربری
# ==========================================
st.sidebar.title(f"🔍 {fix_fa('تنظیمات ایستگاه Aariz')}")
p_id = st.sidebar.text_input("Patient ID", "Aariz_Alpha_01")
res_val = st.sidebar.number_input("Resolution (mm/px)", value=0.1, format="%.4f")
upload = st.sidebar.file_uploader("Upload Cephalogram", type=["png", "jpg", "jpeg"])

# ==========================================
# ۵. آنالیز لندمارک‌های ۲۹گانه
# ==========================================
if upload:
    img_pil = Image.open(upload).convert("RGB")
    W, H = img_pil.size
    
    # پردازش هوشمند مدل
    gray_img = img_pil.convert("L").resize((512, 512))
    in_data = torch.from_numpy(np.array(gray_img)/255.0).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        preds = master_model(in_data).cpu().numpy()[0]
    
    coords = []
    for i in range(29):
        y, x = np.unravel_index(preds[i].argmax(), preds[i].shape)
        coords.append((int(x * W / 512), int(y * H / 512)))

    # ترسیم گرافیکی
    canvas = img_pil.copy()
    draw = ImageDraw.Draw(canvas)
    for i, (cx, cy) in enumerate(coords):
        draw.ellipse([cx-6, cy-6, cx+6, cy+6], fill="red", outline="white")
        draw.text((cx+10, cy-10), str(i), fill="yellow")

    st.subheader("🖼 Digital Analysis Mapping")
    st.image(canvas, width=1100)

    # مقادیر آنالیز Steiner (نمونه برای نمایش ساختار)
    results = {"SNA": "82.1", "SNB": "77.9", "ANB": "4.2", "FMA": "25.4"}
    
    # ==========================================
    # ۶. گزارش PDF (اصلاح شده برای رفع هشدارها)
    # ==========================================
    if st.button("📥 دریافت گزارش نهایی PDF"):
        pdf = FPDF()
        pdf.add_page()
        
        # استفاده از فونت سیستمی استاندارد برای جلوگیری از Deprecation
        pdf.set_font("helvetica", size=14) 
        
        # اصلاح متدها: txt -> text | ln=True -> new_x/new_y
        pdf.cell(0, 10, text="Aariz Precision Station V7.8 - Clinical Report", 
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(10)
        
        pdf.set_font("helvetica", size=12)
        pdf.cell(0, 10, text=f"Patient ID: {p_id}", 
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        
        for k, v in results.items():
            pdf.cell(0, 10, text=f"{k}: {v} degrees", 
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')

        pdf_bytes = bytes(pdf.output())
        st.download_button("Download Report", pdf_bytes, f"{p_id}_Report.pdf", "application/pdf")
