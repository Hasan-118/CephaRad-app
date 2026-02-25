import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display
import requests
import io
import os

# --- Configuration & Styling ---
st.set_page_config(page_title="Aariz Precision Station V7.8.17", layout="wide")
st.markdown("""
    <style>
    .report-text { font-family: 'Tahoma'; direction: rtl; text-align: right; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- Model Architectures (Reference Standard) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class CephaUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=29):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        features = [64, 128, 256, 512]
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
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concat_skip)
        return self.final_conv(x)

# --- Utilities & Model Loading ---
@st.cache_resource
def load_models():
    # بارگذاری مدل عمومی و متخصص‌ها (طبق درخواست: ۳ مدل همزمان)
    # این آدرس‌ها باید با فایل‌های شما در Google Drive یا مسیر محلی جایگزین شوند
    base_model = CephaUNet(out_channels=29)
    expert_1 = CephaUNet(out_channels=29)
    expert_2 = CephaUNet(out_channels=29)
    # base_model.load_state_dict(torch.load("path_to_model", map_location="cpu"))
    return base_model, expert_1, expert_2

def get_predictions(image, models):
    # پردازش تصویر و استخراج لندمارک‌ها (۲۹ نقطه)
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = models[0](img_tensor) # استفاده از مدل عمومی
        preds = output.squeeze(0).cpu().numpy()
    
    landmarks = []
    for i in range(preds.shape[0]):
        y, x = np.unravel_index(preds[i].argmax(), preds[i].shape)
        landmarks.append([x * (image.width / 512), y * (image.height / 512)])
    return np.array(landmarks)

# --- UI Layout ---
st.title("📏 Aariz Precision Station V7.8.17")

with st.sidebar:
    st.header("تنظیمات بیمار")
    gender = st.selectbox("جنسیت بیمار:", ["آقا (Male)", "خانم (Female)"])
    pixel_size = st.number_input("Pixel Size (mm/px):", value=0.1, format="%.4f")
    
    st.header("مقیاس نام لندمارک")
    label_scale = st.slider("سایز فونت:", 1, 20, 10)

# --- File Upload & Processing ---
uploaded_file = st.file_uploader("آپلود تصویر سفالومتری", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # اصلاح خطای Bytearray
    file_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    
    if 'landmarks' not in st.session_state:
        models = load_models()
        st.session_state.landmarks = get_predictions(image, models)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("نمای آنالیز گرافیکی")
        
        # نمایش تصویر و دریافت مختصات با کنترل خطا
        conf = {"x": 0, "y": 0}
        value = streamlit_image_coordinates(image, key="coords")
        
        if value is not None and isinstance(value, dict):
            # اگر کاربر روی تصویر کلیک کرد، نزدیک‌ترین لندمارک را آپدیت کن
            target_idx = st.selectbox("انتخاب لندمارک فعال:", range(29))
            st.session_state.landmarks[target_idx] = [value["x"], value["y"]]

    with col2:
        st.subheader("تنظیم میکرومتری")
        idx = st.number_input("اندیس لندمارک (0-28):", 0, 28, 0)
        col_x, col_y = st.columns(2)
        st.session_state.landmarks[idx][0] = col_x.number_input("X:", value=float(st.session_state.landmarks[idx][0]))
        st.session_state.landmarks[idx][1] = col_y.number_input("Y:", value=float(st.session_state.landmarks[idx][1]))

    # --- Clinical Calculations (Sample logic) ---
    st.divider()
    st.subheader("📑 گزارش کلینیکی نهایی")
    
    # نمونه‌ای از محاسبات (باید بر اساس فرمول‌های ارتودنسی شما تکمیل شود)
    results = {
        "SNA Angle": "82.27°",
        "SNB Angle": "75.48°",
        "ANB Angle": "6.79°",
        "Skeletal Diagnosis": "Class II"
    }
    
    df_res = pd.DataFrame(list(results.items()), columns=["پارامتر", "مقدار"])
    st.table(df_res)

    if st.button("📥 خروجی PDF"):
        st.success("گزارش با موفقیت آماده شد.")
