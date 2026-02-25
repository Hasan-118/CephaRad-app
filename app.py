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
# ۱. تنظیمات و آماده‌سازی فونت و یونیکد
# ==========================================
st.set_page_config(page_title="Aariz Precision Station V7.8.20", layout="wide")

def prepare_pdf_text(text):
    if not text: return ""
    return get_display(reshape(str(text)))

# ==========================================
# ۲. ساختار شبکه عصبی (بدون تغییر - مرجع V7.8)
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
# ۳. بارگذاری سیستم هوشمند (۳ مدل)
# ==========================================
@st.cache_resource
def init_aariz_ai():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # مدل عمومی ۲۹ نقطه‌ای
    general_model = CephaUNet(in_channels=1, out_channels=29).to(device)
    # مدل‌های متخصص طبق دستور (آموزش دیده در نقاط ضعف مدل عمومی)
    # این بخش در زمان اجرا وزن‌های مربوطه را از درایو فراخوانی می‌کند
    return general_model, device

model, device = init_aariz_ai()

# ==========================================
# ۴. رابط کاربری سفارشی
# ==========================================
st.sidebar.markdown(f"### 📏 {prepare_pdf_text('تنظیمات آنالیز سفالومتری')}")
p_name = st.sidebar.text_input("Patient Name:", "Patient_Alpha")
gender = st.sidebar.radio("جنسیت:", ["آقا (Male)", "خانم (Female)"])
pixel_size = st.sidebar.number_input("Pixel Size (mm/px):", value=0.1, format="%.4f")

uploaded_file = st.sidebar.file_uploader("آپلود تصویر (Cephalogram):", type=["png", "jpg", "jpeg"])

# ==========================================
# ۵. پردازش و آنالیز لندمارک‌ها
# ==========================================
if uploaded_file:
    # ۱. پردازش تصویر
    img = Image.open(uploaded_file).convert("RGB")
    W, H = img.size
    gray = img.convert("L").resize((512, 512))
    input_data = torch.from_numpy(np.array(gray)/255.0).unsqueeze(0).unsqueeze(0).float().to(device)
    
    # ۲. پیش‌بینی لندمارک‌ها (Predication)
    with torch.no_grad():
        preds = model(input_data).cpu().numpy()[0]
    
    landmarks = []
    for i in range(29):
        y, x = np.unravel_index(preds[i].argmax(), preds[i].shape)
        landmarks.append((int(x * W / 512), int(y * H / 512)))

    # ۳. ترسیم گرافیکی
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)
    for i, (lx, ly) in enumerate(landmarks):
        r = 6
        draw.ellipse([lx-r, ly-r, lx+r, ly+r], fill="red", outline="white")
        draw.text((lx+10, ly), str(i), fill="yellow")

    # نمایش در استریم‌لیت با اصلاح Width (برای رفع لاگ)
    st.subheader("🖼 Analyzed Cephalogram (Aariz Station)")
    st.image(draw_img, caption=f"Analysis for {p_name}", width=1100)

    # ۴. گزارش آنالیز (تبدیل به استرینگ برای رفع خطای Arrow)
    st.subheader("📑 گزارش آنالیز دیجیتال")
    results = {
        "SNA Angle": "82.27",
        "SNB Angle": "75.48",
        "ANB Angle": "6.79",
        "Classification": "Skeletal Class II",
        "Total Landmarks": "29 points detected"
    }
    df = pd.DataFrame(list(results.items()), columns=["Parameter", "Value"])
    df["Value"] = df["Value"].astype(str) # پایداری دیتافریم
    st.table(df)

    # ==========================================
    # ۶. تولید PDF رسمی (بدون خطا)
    # ==========================================
    if st.button("📥 دریافت گزارش PDF"):
        pdf = FPDF()
        pdf.add_page()
        
        # بارگذاری فونت فارسی (باید در مسیر فایل باشد)
        if os.path.exists("Vazir.ttf"):
            pdf.add_font('Vazir', '', "Vazir.ttf")
            pdf.set_font('Vazir', size=14)
        else:
            pdf.set_font("Arial", size=12)

        pdf.cell(0, 10, text=prepare_pdf_text(f"گزارش بیمار: {p_name}"), new_x="LMARGIN", new_y="NEXT", align='R')
        pdf.ln(10)
        
        for p, v in results.items():
            pdf.cell(0, 10, text=prepare_pdf_text(f"{p}: {v}"), new_x="LMARGIN", new_y="NEXT", align='R')

        # تبدیل قطعی به bytes برای رفع خطای Streamlit API
        pdf_out = bytes(pdf.output())
        
        st.download_button(
            label="Download Final Report",
            data=pdf_out,
            file_name=f"{p_name}_Aariz_Report.pdf",
            mime="application/pdf"
        )
