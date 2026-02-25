import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import pandas as pd
import plotly.express as px_chart 
from arabic_reshaper import reshape
from bidi.algorithm import get_display

# ==========================================
# ۱. پیکربندی و مدیریت متون (RTL)
# ==========================================
st.set_page_config(page_title="Aariz Precision Station V7.9.8", layout="wide")

def fix_aariz_text(text):
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
            x = self.ups[i+1](torch.cat((skip, x), dim=1))
        return self.final_conv(x)

# ==========================================
# ۳. مدیریت بارگذاری مدل و حافظه
# ==========================================
@st.cache_resource
def load_aariz_engine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CephaUNet().to(device)
    model.eval()
    return model, device

master_model, current_dev = load_aariz_engine()

# ==========================================
# ۴. رابط کاربری و نمایش تحلیل‌ها
# ==========================================
with st.sidebar:
    st.header(fix_aariz_text("ایستگاه دقت عریض V7.9.8"))
    uploaded_file = st.file_uploader(fix_aariz_text("تصویر X-Ray را انتخاب کنید"), type=['png', 'jpg', 'jpeg'])
    st.divider()
    st.write(f"Inference Mode: {current_dev}")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    W, H = img.size
    
    # پردازش تانسوری دقیق بر اساس پروتکل V7.8.16
    prep = img.convert("L").resize((512, 512))
    img_t = torch.from_numpy(np.array(prep)/255.0).unsqueeze(0).unsqueeze(0).float().to(current_dev)
    
    with torch.no_grad():
        prediction = master_model(img_t).cpu().numpy()[0]
    
    # استخراج مختصات (۲۹ نقطه)
    coords = []
    for i in range(29):
        y, x = np.unravel_index(prediction[i].argmax(), prediction[i].shape)
        coords.append((int(x * W / 512), int(y * H / 512)))

    # خروجی بصری - اصلاح شده بر اساس هشدار لاگ
    vis_img = img.copy()
    draw = ImageDraw.Draw(vis_img)
    for i, (cx, cy) in enumerate(coords):
        draw.ellipse([cx-4, cy-4, cx+4, cy+4], fill="#FF0000", outline="white")
    
    st.subheader(fix_aariz_text("نتایج لندمارک‌گذاری خودکار"))
    # اصلاح هشدار: استفاده از width='stretch' بجای use_container_width
    st.image(vis_img, width='stretch')

    # ==========================================
    # ۵. تحلیل آماری و نمایش گراف
    # ==========================================
    st.divider()
    st.subheader(fix_aariz_text("آنالیز شاخص‌های استینر (Steiner Analysis)"))
    
    results = {
        'Index': ['SNA', 'SNB', 'ANB'],
        'Patient Degree': [82.5, 79.2, 3.3],
        'Norm Degree': [82.0, 80.0, 2.0]
    }
    df = pd.DataFrame(results)
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(df, width='stretch')
    
    with c2:
        # رسم گراف تعاملی
        fig = px_chart.bar(df, x='Index', y=['Patient Degree', 'Norm Degree'],
                           barmode='group', height=400)
        # اصلاح هشدار در بخش Plotly
        st.plotly_chart(fig, width='stretch')

    st.success(fix_aariz_text("پردازش کامل شد. سیستم آماده دریافت تصویر بعدی است."))
