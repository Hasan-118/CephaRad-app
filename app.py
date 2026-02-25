import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import pandas as pd
import plotly.express as px
from arabic_reshaper import reshape
from bidi.algorithm import get_display

# ==========================================
# ۱. پیکربندی و استایلینگ (Streamlit 2026 Ready)
# ==========================================
st.set_page_config(page_title="Aariz Precision Station V7.9.0", layout="wide")

def fix_text(text):
    if not text: return ""
    return get_display(reshape(str(text)))

# ==========================================
# ۲. معماری مرجع طلایی V7.8.16 (بدون تغییر)
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
        self.pool = nn.MaxPool2d(2, 2)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, 2, 2))
            self.ups.append(DoubleConv(feature*2, feature))
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i//2]
            if x.shape != skip.shape: x = TF.resize(x, skip.shape[2:])
            x = self.ups[i+1](torch.cat((skip, x), 1))
        return self.final_conv(x)

# ==========================================
# ۳. بارگذاری هوشمند و پردازش
# ==========================================
@st.cache_resource
def load_gold_model():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = CephaUNet().to(dev)
    mdl.eval()
    return mdl, dev

model, device = load_gold_model()

# ==========================================
# ۴. رابط کاربری (UI)
# ==========================================
st.sidebar.title(fix_text("پنل مدیریتی عریض"))
uploaded_file = st.sidebar.file_uploader(fix_text("انتخاب تصویر سفالومتری"), type=['png', 'jpg'])

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert("RGB")
    W, H = raw_img.size
    
    # پردازش تصویر مطابق پارامترهای مرجع
    processed = raw_img.convert("L").resize((512, 512))
    tensor_in = torch.from_numpy(np.array(processed)/255.0).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        pred = model(tensor_in).cpu().numpy()[0]
    
    # استخراج ۲۹ نقطه لندمارک
    points = []
    for i in range(29):
        y, x = np.unravel_index(pred[i].argmax(), pred[i].shape)
        points.append((int(x * W / 512), int(y * H / 512)))

    # رسم گرافیکی روی تصویر
    draw_img = raw_img.copy()
    overlay = ImageDraw.Draw(draw_img)
    for i, (px, py) in enumerate(points):
        overlay.ellipse([px-3, py-3, px+3, py+3], fill="red", outline="white")
    
    st.image(draw_img, caption="Aariz Automated Detection", width='stretch')

    # ==========================================
    # ۵. نمایش گراف‌ها و تحلیل (جدید)
    # ==========================================
    st.divider()
    st.subheader(fix_text("گراف مقایسه‌ای شاخص‌های اسکلتی (Steiner Analysis)"))
    
    # داده‌های نمونه (در نسخه نهایی از توابع هندسی جایگزین می‌شود)
    data = {
        'Metric': ['SNA', 'SNB', 'ANB'],
        'Patient': [84.2, 78.5, 5.7],
        'Norm': [82.0, 80.0, 2.0]
    }
    df = pd.DataFrame(data)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        # نمایش جدول با فرمت جدید
        st.dataframe(df, width='stretch')
    
    with col2:
        # گراف تعاملی برای نمایش در گوشی و سیستم
        fig = px.bar(df, x='Metric', y=['Patient', 'Norm'], barmode='group',
                     color_discrete_map={'Patient': '#FF4B4B', 'Norm': '#1F77B4'})
        st.plotly_chart(fig, use_container_width=True)

    st.success(f"Analysis Complete. Device used: {device}")
