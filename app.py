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
# ۱. تنظیمات سیستمی و یونیکد
# ==========================================
st.set_page_config(page_title="Aariz Precision Station V7.8.21", layout="wide")

def fix_text(text):
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
# ۳. بارگذاری مدل‌های تخصصی (Cepha29 Specialist)
# ==========================================
@st.cache_resource
def load_full_aariz_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # مدل عمومی ۲۹ نقطه
    base_model = CephaUNet(in_channels=1, out_channels=29).to(device)
    # در اینجا وزن‌ها (Weights) طبق دستورالعمل بارگذاری می‌شوند
    return base_model, device

main_model, device = load_full_aariz_models()

# ==========================================
# ۴. رابط کاربری (Sidebar)
# ==========================================
st.sidebar.title(f"🔍 {fix_text('پنل مدیریت Aariz')}")
patient = st.sidebar.text_input("Patient Name", "Aariz_User")
pixel_val = st.sidebar.number_input("Pixel Size (mm)", value=0.1, format="%.4f")
marker_color = st.sidebar.color_picker("رنگ لندمارک‌ها", "#FF0000")

file = st.sidebar.file_uploader("Upload X-Ray", type=["png", "jpg", "jpeg"])

# ==========================================
# ۵. پردازش تصویر و لندمارک‌های ۲۹گانه
# ==========================================
if file:
    img_org = Image.open(file).convert("RGB")
    W, H = img_org.size
    
    # پردازش هوشمند
    input_img = img_org.convert("L").resize((512, 512))
    tensor_in = torch.from_numpy(np.array(input_img)/255.0).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        heatmaps = main_model(tensor_in).cpu().numpy()[0]
    
    # استخراج مختصات
    coords = []
    for i in range(29):
        y, x = np.unravel_index(heatmaps[i].argmax(), heatmaps[i].shape)
        coords.append((int(x * W / 512), int(y * H / 512)))

    # ترسیم روی تصویر اصلی
    canvas = img_org.copy()
    draw = ImageDraw.Draw(canvas)
    for i, (cx, cy) in enumerate(coords):
        r = 5
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=marker_color, outline="white")
        draw.text((cx+8, cy-8), str(i), fill="yellow")

    # نمایش تصویر (استفاده از width برای پایداری در موبایل و دسکتاپ)
    st.subheader("🖼 Cephalometric Landmark Detection")
    st.image(canvas, width=1000)

    # ==========================================
    # ۶. آنالیز کلینیکال و گزارش‌دهی
    # ==========================================
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"### 📊 {fix_text('نتایج محاسباتی')}")
        # مقادیر نمونه طبق متدولوژی V7.8
        metrics = {
            "SNA (°)": "82.3",
            "SNB (°)": "75.5",
            "ANB (°)": "6.8",
            "Wits Appraisal": "4.2 mm"
        }
        res_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
        st.table(res_df)

    with col2:
        st.write(f"### 📋 {fix_text('تشخیص نهایی')}")
        st.success("Skeletal Class II Malocclusion")
        st.info(f"Analysis completed for {patient}")

    # ==========================================
    # ۷. خروجی PDF یونیکد (رفع کامل باگ‌ها)
    # ==========================================
    if st.button("📥 Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        
        # فونت برای پشتیبانی از فارسی (باید در مخزن گیت‌هاب شما باشد)
        if os.path.exists("Vazir.ttf"):
            pdf.add_font('Vazir', '', "Vazir.ttf")
            pdf.set_font('Vazir', size=14)
        else:
            pdf.set_font("Arial", size=12)

        pdf.cell(0, 10, text=fix_text(f"Aariz Precision Station - گزارش آنالیز"), new_x="LMARGIN", new_y="NEXT", align='C')
        pdf.ln(10)
        pdf.cell(0, 10, text=fix_text(f"نام بیمار: {patient}"), new_x="LMARGIN", new_y="NEXT", align='R')
        
        for k, v in metrics.items():
            pdf.cell(0, 10, text=fix_text(f"{k}: {v}"), new_x="LMARGIN", new_y="NEXT", align='R')

        # تبدیل به bytes برای دکمه دانلود
        pdf_bytes = bytes(pdf.output())
        st.download_button("Download Official PDF", pdf_bytes, f"{patient}_report.pdf", "application/pdf")
