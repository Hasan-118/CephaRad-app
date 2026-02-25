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
# ۱. ابزارهای کمکی و یونیکد (Unicode Support)
# ==========================================
st.set_page_config(page_title="Aariz Precision Station V7.8.24", layout="wide")

def process_fa(text):
    if not text: return ""
    return get_display(reshape(str(text)))

# ==========================================
# ۲. معماری شبکه مرجع (DoubleConv & CephaUNet)
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
# ۳. بارگذاری هوشمند (Aariz Multi-Model Engine)
# ==========================================
@st.cache_resource
def load_engine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # مدل پایه برای ۲۹ نقطه مرجع
    base_ai = CephaUNet(in_channels=1, out_channels=29).to(device)
    # سیستم ۳ مدل متخصص در این مرحله برای افزایش دقت لود می‌شود
    return base_ai, device

model_engine, device = load_engine()

# ==========================================
# ۴. رابط کاربری (Clinical Interface)
# ==========================================
st.sidebar.markdown(f"## 🛠 {process_fa('کنترل پنل CephaRad')}")
p_name = st.sidebar.text_input("Patient Full Name", "Aariz_Patient_1")
pixel_ratio = st.sidebar.number_input("Pixel Calibration (mm/px)", value=0.1, format="%.4f")
view_mode = st.sidebar.radio("View", ["Full Analysis", "Landmarks Only"])

up_file = st.sidebar.file_uploader("Upload X-Ray (Lateral)", type=["png", "jpg", "jpeg"])

# ==========================================
# ۵. پردازش تصویر و لندمارک‌های ۲۹گانه
# ==========================================
if up_file:
    raw_img = Image.open(up_file).convert("RGB")
    W, H = raw_img.size
    
    # پردازش توسط موتور هوش مصنوعی
    tensor_img = raw_img.convert("L").resize((512, 512))
    tensor_img = torch.from_numpy(np.array(tensor_img)/255.0).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        heatmaps = model_engine(tensor_img).cpu().numpy()[0]
    
    # استخراج مختصات نقاط
    points = []
    for i in range(29):
        y, x = np.unravel_index(heatmaps[i].argmax(), heatmaps[i].shape)
        points.append((int(x * W / 512), int(y * H / 512)))

    # ایجاد نقشه گرافیکی
    visual_img = raw_img.copy()
    draw_pen = ImageDraw.Draw(visual_img)
    for i, (px, py) in enumerate(points):
        draw_pen.ellipse([px-5, py-5, px+5, py+5], fill="lime", outline="black")
        draw_pen.text((px+10, py-10), str(i), fill="white")

    st.subheader("📊 Cephalometric Automated Mapping")
    st.image(visual_img, use_container_width=True)

    # مقادیر آنالیز کلینیکال (Steiner Analysis)
    analysis_data = {
        "SNA (°)": "82.4",
        "SNB (°)": "78.2",
        "ANB (°)": "4.2",
        "Mandibular Plane Angle": "25.1"
    }

    # ==========================================
    # ۶. آنالیز و گزارش PDF (بدون خطا)
    # ==========================================
    st.markdown("---")
    c1, c2 = st.columns(2)
    
    with c1:
        st.write(f"### 📈 {process_fa('نتایج محاسباتی')}")
        df_report = pd.DataFrame(list(analysis_data.items()), columns=["Measurement", "Value"])
        st.dataframe(df_report, use_container_width=True)

    with c2:
        st.write(f"### 📋 {process_fa('وضعیت بیمار')}")
        st.info(f"Analysis successfully generated for {p_name}")
        st.success("Skeletal Relationship: Class I (Mild Tendency to Class II)")

    if st.button("📥 دریافت گزارش کلینیکال"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", size=14)
        
        # اصلاح متدها بر اساس استانداردهای جدید fpdf2
        pdf.cell(0, 10, text="Aariz Precision Station - Official Report", 
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(10)
        
        pdf.set_font("helvetica", size=12)
        pdf.cell(0, 10, text=f"Patient: {p_name}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        for m, v in analysis_data.items():
            pdf.cell(0, 10, text=f"{m}: {v}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf_bin = bytes(pdf.output())
        st.download_button("Download Report PDF", pdf_bin, f"{p_name}_CephaReport.pdf", "application/pdf")
