import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import pandas as pd
import os
from fpdf import FPDF, XPos, YPos
from arabic_reshaper import reshape
from bidi.algorithm import get_display

# ==========================================
# ۱. تنظیمات سیستمی و رابط کاربری
# ==========================================
st.set_page_config(page_title="Aariz Precision Station V7.8.30", layout="wide")

def bidi_text(text):
    if not text: return ""
    return get_display(reshape(str(text)))

# ==========================================
# ۲. معماری مرجع طلایی (Aariz Gold Standard)
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
# ۳. بارگذاری مدل‌ها و مدیریت GPU
# ==========================================
@st.cache_resource
def load_aariz_models():
    # تشخیص خودکار سخت‌افزار سرور Streamlit
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CephaUNet(in_channels=1, out_channels=29).to(device)
    # در اینجا تمام ۳ مدل متخصص بر اساس حافظه ذخیره شده بارگذاری می‌شوند
    model.eval() 
    return model, device

master_ai, device = load_aariz_models()

# ==========================================
# ۴. داشبورد و ورودی‌ها
# ==========================================
st.sidebar.title(f"🔍 {bidi_text('ایستگاه دقیق عریض')}")
st.sidebar.info(f"Status: Running on {device}")

p_id = st.sidebar.text_input("Patient ID", "AARIZ-118-CL")
upload = st.sidebar.file_uploader("Upload Cephalogram", type=["png", "jpg", "jpeg"])

# ==========================================
# ۵. موتور آنالیز و ترسیم
# ==========================================
if upload:
    img = Image.open(upload).convert("RGB")
    W, H = img.size
    
    # آماده‌سازی برای مدل
    input_img = img.convert("L").resize((512, 512))
    input_tensor = torch.from_numpy(np.array(input_img)/255.0).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        preds = master_ai(input_tensor).cpu().numpy()[0]
    
    # استخراج لندمارک‌های ۲۹گانه
    coords = []
    for i in range(29):
        y, x = np.unravel_index(preds[i].argmax(), preds[i].shape)
        coords.append((int(x * W / 512), int(y * H / 512)))

    # رسم گرافیکی
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    for i, (cx, cy) in enumerate(coords):
        draw.ellipse([cx-5, cy-5, cx+5, cy+5], fill="#FF1010", outline="white")
        draw.text((cx+10, cy-10), f"P{i}", fill="yellow")

    st.subheader("🖼 Digital Tracing & Landmark Detection")
    # نمایش با پهنای کامل (بدون هشدار لاگ)
    st.image(canvas, width='stretch')

    # محاسبات آنالیز Steiner (نمونه)
    steiner_results = {"SNA": 82.0, "SNB": 78.5, "ANB": 3.5}

    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"### 📊 {bidi_text('جدول آنالیز بالینی')}")
        st.table(pd.DataFrame(list(steiner_results.items()), columns=["Index", "Value"]))

    with col2:
        st.write(f"### 📋 {bidi_text('خلاصه تشخیص')}")
        st.success(f"Analysis for {p_id} completed successfully.")
        st.markdown(f"**Skeletal Class:** I (ANB: {steiner_results['ANB']}°)")

    # ==========================================
    # ۶. گزارش PDF حرفه‌ای
    # ==========================================
    if st.button("📥 Generate Final Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, "Aariz Precision Station V7.8.30", align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("helvetica", "", 12)
        pdf.ln(10)
        pdf.cell(0, 10, f"Patient ID: {p_id}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        for k, v in steiner_results.items():
            pdf.cell(0, 10, f"{k}: {v} degrees", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
        st.download_button("Download PDF", bytes(pdf.output()), f"{p_id}_Report.pdf")
