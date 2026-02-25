import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import pandas as pd
import os
from fpdf import FPDF
from arabic_reshaper import reshape
from bidi.algorithm import get_display

# ==========================================
# ۱. پیکربندی و توابع کمکی (یونیکد)
# ==========================================
st.set_page_config(page_title="Aariz Precision Station V7.8.17", layout="wide")

def prepare_pdf_text(text):
    if not text: return ""
    return get_display(reshape(str(text)))

# ==========================================
# ۲. معماری شبکه (بدون تغییر - مرجع V7.8.16)
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
# ۳. بارگذاری مدل‌ها و ابزارهای آنالیز
# ==========================================
@st.cache_resource
def load_assets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CephaUNet(in_channels=1, out_channels=29).to(device)
    # در اینجا باید منطق load_state_dict شما قرار بگیرد
    return model, device

model, device = load_assets()

# ==========================================
# ۴. رابط کاربری و پردازش تصویر
# ==========================================
st.sidebar.title("📏 تنظیمات آنالیز")
p_name = st.sidebar.text_input("Patient Name:", "Unnamed")
gender = st.sidebar.radio("جنسیت:", ["آقا (Male)", "خانم (Female)"])
pixel_size = st.sidebar.number_input("Pixel Size (mm/px):", value=0.1, format="%.4f")
uploaded_file = st.sidebar.file_uploader("آپلود تصویر سفالومتری:", type=["png", "jpg", "jpeg"])

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")
    gray_img = raw_img.convert("L")
    w, h = raw_img.size
    
    # پیش‌بینی هوشمند
    img_input = np.array(gray_img.resize((512, 512))) / 255.0
    img_tensor = torch.from_numpy(img_input).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        # استخراج مختصات (فرضی برای نمایش کامل کد)
        landmarks = {i: (np.random.randint(0, w), np.random.randint(0, h)) for i in range(29)}

    # ==========================================
    # ۵. محاسبات کلینیکال (نمونه مرجع)
    # ==========================================
    # محاسبات زوایا (SNA, SNB, ANB) طبق مختصات لندمارک‌ها
    # مثال: SNA = 82.27, SNB = 75.48, ANB = 6.79
    analysis_results = {
        "SNA Angle": 82.27,
        "SNB Angle": 75.48,
        "ANB Angle": 6.79,
        "Classification": "Skeletal Class II"
    }

    # ==========================================
    # ۶. ترسیم آنالیز و نمایش (بخش گرافیکی کامل)
    # ==========================================
    st.subheader("🖼 ترسیم آنالیز و لندمارک‌ها")
    draw = ImageDraw.Draw(raw_img)
    
    # رسم لندمارک‌ها و خطوط آنالیز
    for i, (lx, ly) in landmarks.items():
        radius = 5
        draw.ellipse([lx-radius, ly-radius, lx+radius, ly+radius], fill="red", outline="white")
    
    # رسم خطوط پایه (مثلاً خط N-S)
    if 0 in landmarks and 1 in landmarks: # فرض: 0=Nasion, 1=Sella
        draw.line([landmarks[0], landmarks[1]], fill="yellow", width=3)

    st.image(raw_img, caption="Analyzed Cephalogram", use_container_width=True)

    # نمایش جدول نتایج در اپلیکیشن
    st.subheader("📑 گزارش آنالیز")
    df = pd.DataFrame(list(analysis_results.items()), columns=["Parameter", "Value"])
    st.table(df)

    # ==========================================
    # ۷. خروجی PDF (نسخه نهایی و بدون خطا)
    # ==========================================
    if st.button("📥 Generate & Download PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        
        font_path = "Vazir.ttf"
        if os.path.exists(font_path):
            pdf.add_font('Vazir', '', font_path)
            pdf.set_font('Vazir', size=12)
        else:
            pdf.set_font('Arial', size=12)

        # محتوای گزارش
        pdf.cell(0, 10, txt=prepare_pdf_text("گزارش آنالیز دیجیتال سفالومتری"), new_x="LMARGIN", new_y="NEXT", align='C')
        pdf.ln(10)
        pdf.cell(0, 10, txt=prepare_pdf_text(f"بیمار: {p_name}"), new_x="LMARGIN", new_y="NEXT", align='R')
        pdf.cell(0, 10, txt=prepare_pdf_text(f"جنسیت: {gender}"), new_x="LMARGIN", new_y="NEXT", align='R')
        pdf.ln(5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        for param, val in analysis_results.items():
            line = f"{param}: {val}"
            pdf.cell(0, 10, txt=prepare_pdf_text(line), new_x="LMARGIN", new_y="NEXT", align='R')

        pdf_bytes = pdf.output()
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"Report_{p_name}.pdf",
            mime="application/pdf"
        )
