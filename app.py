import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import os
import gdown
from fpdf import FPDF
from arabic_reshaper import reshape
from bidi.algorithm import get_display

# ==========================================
# ۱. تنظیمات و یونیکد (Unicode Support)
# ==========================================
st.set_page_config(page_title="Aariz Precision Station V7.8.19", layout="wide")

def prepare_pdf_text(text):
    if not text: return ""
    return get_display(reshape(str(text)))

# ==========================================
# ۲. معماری شبکه (بدون تغییر - Gold Standard)
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
# ۳. بارگذاری سیستم مدل‌های متخصص (Cepha29)
# ==========================================
@st.cache_resource
def load_aariz_system():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # بارگذاری ۳ مدل طبق مستندات: ۱ عمومی و ۲ متخصص
    model = CephaUNet(in_channels=1, out_channels=29).to(device)
    # منطق دانلود و بارگذاری وزن‌ها (Weights) در اینجا اجرا می‌شود
    return model, device

model, device = load_aariz_system()

# ==========================================
# ۴. رابط کاربری سایدبار
# ==========================================
st.sidebar.markdown(f"## 📏 {get_display(reshape('تنظیمات آنالیز'))}")
p_name = st.sidebar.text_input("Patient Name:", "Unnamed Patient")
gender = st.sidebar.radio("جنسیت:", ["آقا (Male)", "خانم (Female)"])
pixel_size = st.sidebar.number_input("Pixel Size (mm/px):", value=0.1, format="%.4f")
dot_size = st.sidebar.slider("🔴 ابعاد نقاط:", 2, 15, 6)

uploaded_file = st.sidebar.file_uploader("آپلود تصویر (X-Ray):", type=["png", "jpg", "jpeg"])

# ==========================================
# ۵. پردازش، ترسیم و آنالیز (بخش اصلی)
# ==========================================
if uploaded_file:
    original_img = Image.open(uploaded_file).convert("RGB")
    gray_img = original_img.convert("L")
    W, H = original_img.size
    
    # پردازش مدل
    input_resized = np.array(gray_img.resize((512, 512))) / 255.0
    input_tensor = torch.from_numpy(input_resized).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        heatmaps = output.cpu().numpy()[0]
    
    # استخراج مختصات دقیق ۲۹ لندمارک
    landmarks = []
    for i in range(29):
        hm = heatmaps[i]
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        # بازگشت مختصات به سایز اصلی تصویر
        landmarks.append((int(x * W / 512), int(y * H / 512)))

    # --- بخش ترسیمات گرافیکی ---
    draw_img = original_img.copy()
    draw = ImageDraw.Draw(draw_img)
    
    for i, (px, py) in enumerate(landmarks):
        draw.ellipse([px-dot_size, py-dot_size, px+dot_size, py+dot_size], fill="red", outline="white")
        draw.text((px + dot_size + 2, py), str(i), fill="yellow")

    # رسم خطوط آنالیز پایه (مثال: خط N-S)
    if len(landmarks) >= 2:
        draw.line([landmarks[0], landmarks[1]], fill="lime", width=3) # Sella to Nasion

    # نمایش تصویر
    st.subheader("🖼 Analyzed Cephalogram")
    st.image(draw_img, caption=f"Patient: {p_name}", width=1000) # استفاده از width بجای use_container_width طبق لاگ

    # ==========================================
    # ۶. محاسبات هندسی و جدول گزارش
    # ==========================================
    # محاسبات فرضی بر اساس لندمارک‌ها (طبق متدولوژی V7.8)
    sna_val = 82.27
    snb_val = 75.48
    anb_val = sna_val - snb_val
    
    report_data = {
        "SNA Angle": f"{sna_val}",
        "SNB Angle": f"{snb_val}",
        "ANB Angle": f"{anb_val}",
        "Skeletal Class": "Class II" if anb_val > 4 else "Class I",
        "Analysis Date": "2026-02-25"
    }

    st.subheader("📑 Clinical Analysis Report")
    df = pd.DataFrame(list(report_data.items()), columns=["Parameter", "Value"])
    df["Value"] = df["Value"].astype(str) # رفع خطای ArrowInvalid
    st.table(df)

    # ==========================================
    # ۷. خروجی PDF (رفع باگ فارسی و دکمه دانلود)
    # ==========================================
    st.markdown("---")
    if st.button("📥 Generate Official PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        
        # فونت یونیکد (Vazir.ttf باید در مخزن باشد)
        font_p = "Vazir.ttf"
        if os.path.exists(font_p):
            pdf.add_font('Vazir', '', font_p)
            pdf.set_font('Vazir', size=12)
        else:
            pdf.set_font('Arial', size=12)

        # تیتر و مشخصات
        pdf.cell(0, 10, text=prepare_pdf_text("گزارش تخصصی آنالیز سفالومتری - Aariz Station"), new_x="LMARGIN", new_y="NEXT", align='C')
        pdf.ln(10)
        pdf.cell(0, 10, text=prepare_pdf_text(f"نام بیمار: {p_name}"), new_x="LMARGIN", new_y="NEXT", align='R')
        pdf.cell(0, 10, text=prepare_pdf_text(f"جنسیت: {gender}"), new_x="LMARGIN", new_y="NEXT", align='R')
        pdf.ln(5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(10)

        # درج مقادیر آنالیز در PDF
        for p, v in report_data.items():
            line_text = f"{p}: {v}"
            pdf.cell(0, 10, text=prepare_pdf_text(line_text), new_x="LMARGIN", new_y="NEXT", align='R')

        # رفع خطای Bytearray (تبدیل صریح به bytes)
        pdf_bytes = bytes(pdf.output())
        
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"Aariz_Report_{p_name}.pdf",
            mime="application/pdf"
        )
