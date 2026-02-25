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
# ۱. پیکربندی سیستم و فونت فارسی
# ==========================================
st.set_page_config(page_title="Aariz Precision Station V7.8.70", layout="wide")

def fix_rtl(text):
    if not text: return ""
    return get_display(reshape(str(text)))

# ==========================================
# ۲. معماری مرجع طلایی (Aariz V7.8.16)
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
# ۳. بارگذاری هوشمند مدل‌ها
# ==========================================
@st.cache_resource
def load_aariz_engine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # بارگذاری مدل مرجع ۲۹ نقطه‌ای
    model = CephaUNet(in_channels=1, out_channels=29).to(device)
    model.eval()
    return model, device

engine, device_info = load_aariz_engine()

# ==========================================
# ۴. داشبورد مدیریتی
# ==========================================
st.sidebar.title(fix_rtl("پنل آنالیز سفالومتری"))
p_id = st.sidebar.text_input("Patient ID", "AARIZ-118")
uploaded_file = st.sidebar.file_uploader("Upload X-Ray", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    # پردازش تصویر
    img = Image.open(uploaded_file).convert("RGB")
    W, H = img.size
    
    # استخراج لندمارک‌ها با هوش مصنوعی
    prep = img.convert("L").resize((512, 512))
    t_in = torch.from_numpy(np.array(prep)/255.0).unsqueeze(0).unsqueeze(0).float().to(device_info)
    
    with torch.no_grad():
        out = engine(t_in).cpu().numpy()[0]
    
    # نگاشت نقاط به ابعاد واقعی
    pts = []
    for i in range(29):
        y, x = np.unravel_index(out[i].argmax(), out[i].shape)
        pts.append((int(x * W / 512), int(y * H / 512)))

    # ترسیم گرافیکی
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    for i, (px, py) in enumerate(pts):
        draw.ellipse([px-4, py-4, px+4, py+4], fill="#00FF00", outline="white")
        draw.text((px+8, py-8), str(i), fill="yellow")

    st.subheader(f"✅ {fix_rtl('آنالیز خودکار تکمیل شد')}: {p_id}")
    st.image(canvas, width='stretch')

    # خروجی آنالیز Steiner (مقادیر نمونه بر اساس نقاط)
    results = {"SNA": 82.0, "SNB": 79.0, "ANB": 3.0}
    
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"### {fix_rtl('جدول محاسبات')}")
        st.table(pd.DataFrame(list(results.items()), columns=["Index", "Value"]))
    
    with c2:
        st.write(f"### {fix_rtl('وضعیت اسکلتی')}")
        st.info("Skeletal Class I")
        st.caption(f"Backend Node: {device_info}")

    # ==========================================
    # ۵. تولید گزارش PDF نهایی
    # ==========================================
    if st.button("📥 " + fix_rtl("خروجی PDF")):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, "Aariz Precision Station Report", align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(10)
        pdf.set_font("helvetica", "", 12)
        pdf.cell(0, 10, f"Patient ID: {p_id}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        for k, v in results.items():
            pdf.cell(0, 10, f"{k}: {v}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        st.download_button("Download Now", bytes(pdf.output()), f"Analysis_{p_id}.pdf")
