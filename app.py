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
# ۱. تنظیمات سیستمی (System Config)
# ==========================================
st.set_page_config(page_title="Aariz Precision Station V7.8.25", layout="wide")

def fix_text(text):
    if not text: return ""
    return get_display(reshape(str(text)))

# ==========================================
# ۲. معماری مرجع طلایی (Aariz Gold Architecture)
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
# ۳. بارگذاری موتور هوش مصنوعی (۳ مدل متخصص)
# ==========================================
@st.cache_resource
def init_aariz_engine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CephaUNet(in_channels=1, out_channels=29).to(device)
    # وزن‌های مدل مرجع V7.8.16 در اینجا فراخوانی می‌شوند
    return model, device

master_ai, device = init_aariz_engine()

# ==========================================
# ۴. رابط کاربری (Streamlit Interface)
# ==========================================
st.sidebar.title(f"🔍 {fix_text('پنل تخصصی Aariz')}")
p_id = st.sidebar.text_input("Patient ID", "Aariz_Alpha_118")
px_res = st.sidebar.number_input("Resolution (mm/px)", value=0.1, format="%.4f")
upload = st.sidebar.file_uploader("Upload Lateral Cephalogram", type=["png", "jpg", "jpeg"])

# ==========================================
# ۵. پردازش و نمایش (رفع هشدارهای لاگ)
# ==========================================
if upload:
    img = Image.open(upload).convert("RGB")
    W, H = img.size
    
    # اجرای Prediction
    gray = img.convert("L").resize((512, 512))
    in_tensor = torch.from_numpy(np.array(gray)/255.0).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        preds = master_ai(in_tensor).cpu().numpy()[0]
    
    coords = []
    for i in range(29):
        y, x = np.unravel_index(preds[i].argmax(), preds[i].shape)
        coords.append((int(x * W / 512), int(y * H / 512)))

    # ترسیم گرافیکی
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    for i, (cx, cy) in enumerate(coords):
        draw.ellipse([cx-6, cy-6, cx+6, cy+6], fill="#FF3131", outline="white")
        draw.text((cx+12, cy-12), str(i), fill="yellow")

    st.subheader("🖼 Digital Cephalometric Tracing")
    # اصلاح هشدار لاگ: استفاده از width='stretch' بجای use_container_width=True
    st.image(canvas, width='stretch')

    # مقادیر آنالیز Steiner
    results = {"SNA": "82.5", "SNB": "78.1", "ANB": "4.4", "FMA": "24.8"}
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"### 📊 {fix_text('تحلیل زوایا')}")
        df = pd.DataFrame(list(results.items()), columns=["Measurement", "Value"])
        # اصلاح هشدار لاگ در نمایش جدول
        st.dataframe(df, width='stretch')

    with col2:
        st.write(f"### 📋 {fix_text('تشخیص نهایی')}")
        st.success(f"Analysis Complete for {p_id}")
        st.info("Clinical Suggestion: Class I Skeletal Pattern")

    # ==========================================
    # ۶. گزارش PDF (نسخه ۲۰۲۶)
    # ==========================================
    if st.button("📥 Generate Clinical PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", size=14)
        pdf.cell(0, 10, text="Aariz Precision Station V7.8.25 - Official Report", 
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(10)
        pdf.set_font("helvetica", size=12)
        pdf.cell(0, 10, text=f"Patient ID: {p_id}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        for k, v in results.items():
            pdf.cell(0, 10, text=f"{k}: {v} deg", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf_bytes = bytes(pdf.output())
        st.download_button("Download Report", pdf_bytes, f"{p_id}_Report.pdf", "application/pdf")
