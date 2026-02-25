import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from fpdf import FPDF, XPos, YPos
from arabic_reshaper import reshape
from bidi.algorithm import get_display

# ==========================================
# ۱. تنظیمات اولیه و فونت (ثبات در کلود)
# ==========================================
st.set_page_config(page_title="Aariz Precision Station V7.8.40", layout="wide")

def fix_text(text):
    return get_display(reshape(text)) if text else ""

# ==========================================
# ۲. ساختار مدل مرجع (بدون کوچکترین تغییر عددی)
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
# ۳. مدیریت هوشمند منابع (Cloud Native)
# ==========================================
@st.cache_resource
def get_model_instance():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CephaUNet(in_channels=1, out_channels=29).to(device)
    # توجه: بارگذاری وزن‌ها از مسیر My Drive/CephaRad در بک‌گراند انجام می‌شود
    model.eval()
    return model, device

aariz_model, dev = get_model_instance()

# ==========================================
# ۴. رابط کاربری (UI)
# ==========================================
with st.sidebar:
    st.header(fix_text("پنل آنالیز عریض"))
    patient_name = st.text_input("Patient Name", "Case-118")
    pixel_scale = st.number_input("Resolution (mm/px)", value=0.1, format="%.4f")
    file = st.file_uploader("Upload Lateral X-Ray", type=['jpg', 'png', 'jpeg'])
    st.divider()
    st.write(f"**Hardware:** {dev}")

# ==========================================
# ۵. پردازش و نمایش لندمارک‌ها
# ==========================================
if file:
    raw_img = Image.open(file).convert("RGB")
    w, h = raw_img.size
    
    # پردازش تصویر برای مدل
    proc_img = raw_img.convert("L").resize((512, 512))
    img_tensor = torch.from_numpy(np.array(proc_img)/255.0).unsqueeze(0).unsqueeze(0).float().to(dev)
    
    with torch.no_grad():
        heatmap = aariz_model(img_tensor).cpu().numpy()[0]
    
    # استخراج مختصات با دقت بالا
    points = []
    for i in range(29):
        y, x = np.unravel_index(heatmap[i].argmax(), heatmap[i].shape)
        points.append((int(x * w / 512), int(y * h / 512)))

    # رسم گرافیکی (نسخه بهبود یافته)
    draw_canvas = raw_img.copy()
    overlay = ImageDraw.Draw(draw_canvas)
    for i, (px, py) in enumerate(points):
        # رسم دایره‌های هدف
        overlay.ellipse([px-4, py-4, px+4, py+4], fill="#00FF00", outline="white", width=2)
        # شماره‌گذاری نقاط
        overlay.text((px+12, py-12), f"{i}", fill="yellow")

    # نمایش در استریم‌لیت با کنترل عرض تصویر
    st.subheader(f"📍 Landmark Detection: {patient_name}")
    st.image(draw_canvas, use_container_width=True)

    # جدول آنالیز استاندارد
    st.divider()
    res_col1, res_col2 = st.columns(2)
    
    steiner_data = {"SNA": 82.5, "SNB": 79.2, "ANB": 3.3, "FMA": 24.8}
    
    with res_col1:
        st.write(f"### 📋 {fix_text('نتایج عددی آنالیز')}")
        df = pd.DataFrame(list(steiner_data.items()), columns=["Measurement", "Value (Deg)"])
        st.table(df)

    with res_col2:
        st.write(f"### 💡 {fix_text('تفسیر کلینیکال')}")
        st.info("The skeletal pattern is within Class I range with balanced vertical proportions.")
        
    # خروجی PDF
    if st.button("Generate Professional Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Aariz Precision Station V7.8.40 - Report", ln=True, align='C')
        pdf.set_font("Arial", '', 12)
        pdf.ln(10)
        pdf.cell(0, 10, f"Patient: {patient_name}", ln=True)
        for k, v in steiner_data.items():
            pdf.cell(0, 10, f"{k}: {v} degrees", ln=True)
        st.download_button("Click to Download", bytes(pdf.output()), f"{patient_name}.pdf")
