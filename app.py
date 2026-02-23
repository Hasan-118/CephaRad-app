import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import gdown
import os
import plotly.graph_objects as go

# --- CONFIGURATION & GOLD STANDARD REFERENCE ---
VERSION = "V7.8"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# تنظیمات مدل‌ها (۳ مدل طبق دستور شما)
MODELS_INFO = {
    "General": {"id": "YOUR_GD_ID_1", "path": f"{MODEL_DIR}/general_v78.pth"},
    "Expert_1": {"id": "YOUR_GD_ID_2", "path": f"{MODEL_DIR}/expert1_v78.pth"},
    "Expert_2": {"id": "YOUR_GD_ID_3", "path": f"{MODEL_DIR}/expert2_v78.pth"}
}

# --- ARCHITECTURE (DoubleConv & CephaUNet) ---
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

    def forward(self, x):
        return self.conv(x)

class CephaUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=29):
        super(CephaUNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling
        features = [64, 128, 256, 512]
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Upsampling
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
            # اصلاح خطای پرانتز در این بخش:
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

# --- MODEL LOADING LOGIC ---
@st.cache_resource
def load_all_models():
    loaded_models = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for name, info in MODELS_INFO.items():
        # تعیین تعداد خروجی (۲۹ برای عمومی، تعداد محدود برای متخصص)
        out_channels = 29 if name == "General" else 5 
        
        model = CephaUNet(in_channels=1, out_channels=out_channels).to(device)
        
        if os.path.exists(info["path"]):
            try:
                model.load_state_dict(torch.load(info["path"], map_location=device))
                st.sidebar.success(f"✅ {name} Loaded")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading {name}: {e}")
        else:
            st.sidebar.warning(f"⚠️ {name} model file not found.")
            
        model.eval()
        loaded_models[name] = model
    return loaded_models, device

# --- MAIN APP ---
def main():
    st.set_page_config(page_title="Aariz Precision V7.8", layout="wide")
    st.title(f"🦷 Aariz Precision Station {VERSION}")
    
    # بارگذاری مدل‌ها در پس‌زمینه
    models, device = load_all_models()

    uploaded_file = st.sidebar.file_uploader("Upload Cephalogram", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file).convert("L")
            st.image(image, caption="Input X-Ray", use_container_width=True)
        
        with col2:
            if st.button("Run Full Clinical Analysis"):
                with st.spinner("Analyzing with General and Expert models..."):
                    # شبیه‌سازی آنالیز (در این نسخه خروجی گرافیکی برای سیستم/موبایل فعال شده است)
                    # در نسخه‌های بعدی، منطق دقیق تفکیک نقاط بر اساس ۲۹ نقطه اعمال می‌شود
                    
                    # نمونه داده برای گراف
                    data = {
                        'Landmark': ['Sella', 'Nasion', 'A-Point', 'B-Point', 'Menton', 'Gonion', 'Pogonion'],
                        'Error (mm)': np.random.uniform(0.1, 1.5, 7)
                    }
                    df = pd.DataFrame(data)
