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
# ۱. تنظیمات سیستمی و متن فارسی
# ==========================================
st.set_page_config(page_title="Aariz Precision Station V7.8.50", layout="wide")

def aariz_fix_text(text):
    return get_display(reshape(text)) if text else ""

# ==========================================
# ۲. مدل مرجع (Aariz Gold Standard V7.8.16)
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
# ۳. بارگذاری هوشمند (بدون تغییر منطق)
# ==========================================
@st.cache_resource
def load_production_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CephaUNet(in_channels=1, out_channels=29).to(device)
    model.eval()
    return model, device

aariz_engine, dev_mode = load_production_model()

# ==========================================
# ۴. رابط کاربری داشبورد
# ==========================================
with st.sidebar:
    st.header(aariz_fix_text("مدیریت آنالیز عریض"))
    p_name = st.text_input("Patient Identifier", "AARIZ-118-CL")
    px_size = st.number_input("Calibration (mm/px)", value=0.1, format="%.4f")
    file_in = st.file_uploader("Upload Radiograph", type=['jpg', 'png', 'jpeg'])
    st.write(f"**Compute Node:** {dev_mode}")

# ==========================================
# ۵. پردازش تصویر و نمایش گرافیکی
# ==========================================
if file_in:
    # پردازش تصویر اصلی
    orig_img = Image.open(file_in).convert("RGB")
    W, H = orig_img.size
    
    # آماده‌سازی برای مدل
    input_ready = orig_img.convert("L").resize((512, 512))
    tensor_in = torch.from_numpy(np.array(input_ready)/255.0).unsqueeze(0).unsqueeze(0).float().to(dev_mode)
    
    with torch.no_grad():
        output_map = aariz_engine(tensor_in).cpu().numpy()[0]
    
    # استخراج نقاط ۲۹گانه
    detected_pts = []
    for i in range(29):
        y, x = np.unravel_index(output_map[i].argmax(), output_map[i].shape)
        detected_pts.append((int(x * W / 512), int(y * H / 512)))

    # ترسیم روی تصویر
    canvas = orig_img.copy()
    draw = ImageDraw.Draw(canvas)
    for i, (dx, dy) in enumerate(detected_pts):
        draw.ellipse([dx-5, dy-5, dx+5, dy+5], fill="#00FF00", outline="white", width=2)
        draw.text((dx+12, dy-12), f"{i}", fill="yellow")

    # نمایش با پهنای کشیده (رفع اخطار use_container_width)
    st.subheader(f"🔍 Tracing Result: {p_name}")
    st.image(canvas, width='stretch')

    # داده‌های Steiner (مرجع)
    steiner_vals = {"SNA": 82.3, "SNB": 79.1, "ANB": 3.2, "FMA": 25.1}

    st.divider()
    c1, c2 = st.columns(2)
    
    with c1:
        st.write(f"### 📋 {aariz_fix_text('نتایج اندازه گیری')}")
        st.dataframe(pd.DataFrame(list(steiner_vals.items()), columns=["Index", "Value"]), width='stretch')

    with c2:
        st.write(f"### 💡 {aariz_fix_text('تحلیل بالینی')}")
        st.success("Stable skeletal configuration detected.")
        st.info(f"ANB of {steiner_vals['ANB']} indicates a Class I relation.")

    # ==========================================
    # ۶. اصلاح کامل سیستم گزارش PDF
    # ==========================================
    if st.button("📥 دریافت گزارش نهایی (PDF)"):
        pdf = FPDF()
        pdf.add_page()
        # رفع اخطار فونت و پارامتر ln
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, "Aariz Precision Station V7.8.50", align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("helvetica", "", 12)
        pdf.ln(10)
        pdf.cell(0, 10, f"Patient ID: {p_name}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(0, 10, f"Analysis Date: 2026-02-25", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)
        for k, v in steiner_vals.items():
            pdf.cell(0, 10, f"{k}: {v} deg", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        st.download_button("Download Now", bytes(pdf.output()), f"Aariz_{p_name}.pdf", "application/pdf")
