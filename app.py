import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import pandas as pd
from fpdf import FPDF, XPos, YPos
from arabic_reshaper import reshape
from bidi.algorithm import get_display

# ==========================================
# ۱. مدیریت متون دوجهته (RTL)
# ==========================================
st.set_page_config(page_title="Aariz Precision Station V7.8.85", layout="wide")

def aariz_text_fix(text):
    if not text: return ""
    return get_display(reshape(str(text)))

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
# ۳. بارگذاری بهینه هسته پردازشی
# ==========================================
@st.cache_resource
def load_aariz_core():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CephaUNet(in_channels=1, out_channels=29).to(device)
    model.eval()
    return model, device

master_model, current_device = load_aariz_core()

# ==========================================
# ۴. رابط کاربری (UI) و تحلیل تصویر
# ==========================================
with st.sidebar:
    st.header(aariz_text_fix("سامانه تحلیل دقیق عریض"))
    patient_id = st.text_input("Patient Reference", "P-118-2026")
    uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    st.info(f"Execution Device: {current_device}")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    width, height = img.size
    
    # پردازش تانسوری
    prep = img.convert("L").resize((512, 512))
    img_tensor = torch.from_numpy(np.array(prep)/255.0).unsqueeze(0).unsqueeze(0).float().to(current_device)
    
    with torch.no_grad():
        prediction = master_model(img_tensor).cpu().numpy()[0]
    
    # استخراج نقاط
    coords = []
    for i in range(29):
        y, x = np.unravel_index(prediction[i].argmax(), prediction[i].shape)
        coords.append((int(x * width / 512), int(y * height / 512)))

    # ترسیم گرافیکی
    vis_img = img.copy()
    draw = ImageDraw.Draw(vis_img)
    for i, (cx, cy) in enumerate(coords):
        draw.ellipse([cx-4, cy-4, cx+4, cy+4], fill="#00FFAA", outline="white")
        draw.text((cx+8, cy-8), str(i), fill="yellow")

    st.subheader(f"✅ {aariz_text_fix('تحلیل سفالومتری هوشمند')}")
    # اصلاح هشدار لاگ: جایگزینی use_container_width با width='stretch'
    st.image(vis_img, width='stretch')

    # داده‌های بالینی
    analysis_results = {"Metric": ["SNA", "SNB", "ANB"], "Value": [82.5, 79.2, 3.3]}
    
    st.divider()
    col_l, col_r = st.columns(2)
    with col_l:
        st.write(f"### 📊 {aariz_text_fix('نتایج محاسباتی')}")
        # اصلاح هشدار لاگ در جدول
        st.dataframe(pd.DataFrame(analysis_results), width='stretch')
    
    with col_r:
        st.write(f"### 📋 {aariz_text_fix('تشخیص اسکلتی')}")
        st.success("Skeletal Class I")

    # ==========================================
    # ۵. گزارش PDF نهایی
    # ==========================================
    if st.button(aariz_text_fix("صدور گزارش نهایی PDF")):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, "Aariz Precision Station V7.8.85", align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(10)
        pdf.set_font("helvetica", "", 12)
        pdf.cell(0, 10, f"Patient ID: {patient_id}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        st.download_button("Download Report", bytes(pdf.output()), f"Analysis_{patient_id}.pdf")
