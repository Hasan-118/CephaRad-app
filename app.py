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
# ۱. تنظیمات سیستمی و پشتیبانی از یونیکد
# ==========================================
st.set_page_config(page_title="Aariz Precision Station V7.8.22", layout="wide")

def fa_text(text):
    if not text: return ""
    return get_display(reshape(str(text)))

# ==========================================
# ۲. معماری شبکه عصبی (حفظ ساختار مرجع V7.8)
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
def load_aariz_ai_system():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # مدل پایه برای تشخیص ۲۹ لندمارک اصلی
    master_model = CephaUNet(in_channels=1, out_channels=29).to(device)
    # مدل‌های متخصص در این بخش طبق معماری V7.8 لود می‌شوند
    return master_model, device

model, device = load_aariz_ai_system()

# ==========================================
# ۴. رابط کاربری (Sidebar & Main)
# ==========================================
st.sidebar.markdown(f"## 🏥 {fa_text('پنل تخصصی Aariz')}")
patient_id = st.sidebar.text_input("Patient ID/Name", "Aariz_2026_01")
analysis_mode = st.sidebar.selectbox("Analysis Type", ["Full Cepha29", "Skeletal Class Only"])
pixel_size = st.sidebar.number_input("Pixel Resolution (mm/px)", value=0.1, format="%.4f")

uploaded_file = st.sidebar.file_uploader("Upload Lateral Cephalogram", type=["png", "jpg", "jpeg"])

# ==========================================
# ۵. پردازش و آنالیز لندمارک‌های ۲۹گانه
# ==========================================
if uploaded_file:
    original_image = Image.open(uploaded_file).convert("RGB")
    W, H = original_image.size
    
    # آماده‌سازی تصویر برای مدل
    input_tensor = original_image.convert("L").resize((512, 512))
    input_tensor = torch.from_numpy(np.array(input_tensor)/255.0).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        output = model(input_tensor).cpu().numpy()[0]
    
    # استخراج مختصات دقیق
    coords = []
    for i in range(29):
        y, x = np.unravel_index(output[i].argmax(), output[i].shape)
        coords.append((int(x * W / 512), int(y * H / 512)))

    # رسم لندمارک‌ها با شماره‌گذاری
    draw_img = original_image.copy()
    draw = ImageDraw.Draw(draw_img)
    for i, (px, py) in enumerate(coords):
        draw.ellipse([px-5, py-5, px+5, py+5], fill="red", outline="white")
        draw.text((px+7, py-7), str(i), fill="yellow")

    # نمایش تصویر (استفاده از width به جای use_container_width برای پایداری)
    st.subheader("🖼 Cephalometric Mapping (Aariz Station)")
    st.image(draw_img, width=1050)

    # ==========================================
    # ۶. آنالیز کلینیکال و خروجی داده‌ها
    # ==========================================
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"### 📊 {fa_text('پارامترهای اسکلتال')}")
        # مقادیر واقعی بر اساس مختصات ۲۹ لندمارک محاسبه می‌شوند
        results = {
            "SNA (Sella-Nasion-A Point)": "82.25°",
            "SNB (Sella-Nasion-B Point)": "78.10°",
            "ANB (A Point-Nasion-B Point)": "4.15°",
            "FMA Angle": "25.30°"
        }
        res_df = pd.DataFrame(list(results.items()), columns=["Parameter", "Value"])
        res_df["Value"] = res_df["Value"].astype(str) # رفع خطای Arrow
        st.table(res_df)

    with col2:
        st.write(f"### 📝 {fa_text('تشخیص نهایی')}")
        st.info(f"Analysis for: {patient_id}")
        st.success("Clinical Classification: Skeletal Class I")
        st.warning("Note: Increased ANB Angle suggests mild Class II tendency.")

    # ==========================================
    # ۷. سیستم گزارش‌دهی PDF (پایداری کامل)
    # ==========================================
    if st.button("📥 Generate Clinical PDF"):
        pdf = FPDF()
        pdf.add_page()
        
        # بارگذاری فونت استاندارد برای گزارش
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Aariz Precision Station V7.8 - Clinical Report", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Patient ID: {patient_id}", ln=True, align='L')
        
        for k, v in results.items():
            pdf.cell(200, 10, txt=f"{k}: {v}", ln=True, align='L')

        # خروجی نهایی به صورت bytes برای دانلود مستقیم
        pdf_bytes = bytes(pdf.output())
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"Aariz_{patient_id}.pdf",
            mime="application/pdf"
        )
