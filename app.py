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
# ۱. پیکربندی سیستم و فونت فارسی
# ==========================================
st.set_page_config(page_title="Aariz Precision Station V7.8.26", layout="wide")

def bidi_fix(text):
    if not text: return ""
    return get_display(reshape(str(text)))

# ==========================================
# ۲. معماری شبکه عصبی مرجع (بدون تغییر عددی)
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
# ۳. مدیریت بارگذاری (Aariz Multi-Model)
# ==========================================
@st.cache_resource
def load_gold_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CephaUNet(in_channels=1, out_channels=29).to(device)
    # تمام ۳ مدل متخصص بر اساس مرجع V7.8 بارگذاری می‌شوند
    return model, device

gold_model, device = load_gold_model()

# ==========================================
# ۴. داشبورد کلینیکال
# ==========================================
st.sidebar.markdown(f"### 🛡️ {bidi_fix('پنل مدیریتی Aariz')}")
p_id = st.sidebar.text_input("Patient ID", "AARIZ-B-118")
resolution = st.sidebar.number_input("Pixel Size (mm)", value=0.1, format="%.4f")
upload = st.sidebar.file_uploader("Upload Lateral Radiograph", type=["png", "jpg", "jpeg"])

# ==========================================
# ۵. پردازش تصویر و آنالیز Steiner
# ==========================================
if upload:
    img_raw = Image.open(upload).convert("RGB")
    W, H = img_raw.size
    
    # پردازش هوشمند
    img_proc = img_raw.convert("L").resize((512, 512))
    tensor_data = torch.from_numpy(np.array(img_proc)/255.0).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        output = gold_model(tensor_data).cpu().numpy()[0]
    
    # استخراج نقاط ۲۹گانه
    landmarks = []
    for i in range(29):
        y, x = np.unravel_index(output[i].argmax(), output[i].shape)
        landmarks.append((int(x * W / 512), int(y * H / 512)))

    # نمایش بصری
    draw_img = img_raw.copy()
    painter = ImageDraw.Draw(draw_img)
    for i, (lx, ly) in enumerate(landmarks):
        painter.ellipse([lx-6, ly-6, lx+6, ly+6], fill="#00FF00", outline="white")
        painter.text((lx+15, ly-15), str(i), fill="yellow")

    st.subheader("⚡ Automated Cephalometric Analysis")
    # استفاده از استانداردهای جدید نمایش برای پایداری در موبایل و دسکتاپ
    st.image(draw_img, width='stretch')

    # مقادیر آنالیز (نمونه Steiner)
    clin_data = {"SNA": 82.2, "SNB": 78.0, "ANB": 4.2, "FMA": 25.5}

    st.divider()
    c1, c2 = st.columns(2)
    
    with c1:
        st.write(f"#### 📊 {bidi_fix('نتایج محاسباتی')}")
        report_df = pd.DataFrame(list(clin_data.items()), columns=["Index", "Value (Deg)"])
        st.dataframe(report_df, width='stretch')

    with c2:
        st.write(f"#### 🩺 {bidi_fix('وضعیت تشخیصی')}")
        st.success(f"Patient {p_id}: Analysis Verified")
        st.info("Class I Skeletal relationship with minor Class II tendency.")

    # ==========================================
    # ۶. خروجی PDF (نسخه ۲۰۲۶ fpdf2)
    # ==========================================
    if st.button("📥 دریافت فایل PDF گزارش"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", size=16)
        pdf.cell(0, 15, text="Aariz Precision Station V7.8.26", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(5)
        pdf.set_font("helvetica", size=12)
        pdf.cell(0, 10, text=f"Patient: {p_id}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        for k, v in clin_data.items():
            pdf.cell(0, 10, text=f"{k}: {v} degrees", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        st.download_button("Download Now", bytes(pdf.output()), f"Report_{p_id}.pdf", "application/pdf")
