import os
import gdown
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۰. تابع دانلود مدل‌ها (قبل از لود کردن مدل‌ها باید اجرا شود) ---
def download_models():
    model_ids = {
        'checkpoint_unet_clinical.pth': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks',
        'specialist_pure_model.pth': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU',
        'tmj_specialist_model.pth': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'
    }
    
    for filename, file_id in model_ids.items():
        if not os.path.exists(filename):
            with st.spinner(f'در حال دانلود مدل {filename} از گوگل درایو...'):
                url = f'https://drive.google.com/uc?id={file_id}'
                try:
                    gdown.download(url, filename, quiet=False)
                except Exception as e:
                    st.error(f"خطا در دانلود {filename}: {e}")

# --- ۱. معماری مدل (بدون تغییر) ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_prob=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class CephaUNet(nn.Module):
    def __init__(self, n_landmarks=29):
        super().__init__()
        self.inc = DoubleConv(1, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512, dropout_prob=0.3))
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up1 = DoubleConv(512, 256, dropout_prob=0.3)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_landmarks, kernel_size=1)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3)
        x = self.up1(x4); x = torch.cat([x, x3], dim=1); x = self.conv_up1(x)
        x = self.up2(x); x = torch.cat([x, x2], dim=1); x = self.conv_up2(x)
        x = self.up3(x); x = torch.cat([x, x1], dim=1); x = self.conv_up3(x)
        return self.outc(x)

# --- ۲. لودر مدل‌ها و پیش‌بینی ---
@st.cache_resource
def load_all_engines():
    # اول دانلود مدل‌ها اگر وجود ندارند
    download_models()
    
    paths = ['checkpoint_unet_clinical.pth', 'specialist_pure_model.pth', 'tmj_specialist_model.pth']
    models = []
    for p in paths:
        if os.path.exists(p):
            try:
                m = CephaUNet(n_landmarks=29)
                ckpt = torch.load(p, map_location="cpu")
                state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
                m.load_state_dict(state)
                m.eval()
                models.append(m)
            except Exception as e:
                st.warning(f"مدل {p} بارگذاری نشد: {e}")
    return models

def run_inference(image_pil, models):
    img_np = np.array(image_pil.convert('L'))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_np)
    oh, ow = img_enhanced.shape
    img_res = cv2.resize(img_enhanced, (384, 384))
    input_t = transforms.ToTensor()(img_res).unsqueeze(0)
    
    hms = []
    with torch.no_grad():
        for m in models: hms.append(m(input_t)[0].numpy())
    
    avg_hm = np.mean(hms, axis=0)
    lms = {}
    for i in range(29):
        y, x = np.unravel_index(np.argmax(avg_hm[i]), (384,384))
        lms[i] = [int(x * ow / 384), int(y * oh / 384)]
    return lms, (ow, oh)

# --- ۳. رابط کاربری (UI) ---
st.set_page_config(layout="wide", page_title="Aariz AI Mobile")
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

# فراخوانی لودر (که خودش دانلودر را صدا می‌زند)
engines = load_all_engines()

with st.sidebar:
    st.header("📲 Aariz Control")
    ui_width = st.slider("Magnification Scale", 300, 1200, 750)
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    target_idx = st.selectbox("Landmark", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")

if uploaded_file and engines:
    img_raw = Image.open(uploaded_file).convert("RGB")
    file_id = uploaded_file.name
    
    if "lms" not in st.session_state or st.session_state.get("file_id") != file_id:
        with st.spinner("Analyzing..."):
            st.session_state.lms, st.session_state.orig_size = run_inference(img_raw, engines)
            st.session_state.file_id = file_id

    col_main, col_detail = st.columns([2, 1])
    
    with col_main:
        ow, oh = st.session_state.orig_size
        draw = ImageDraw.Draw(img_raw)
        try: font = ImageFont.truetype("arial.ttf", int(ow * 0.03))
        except: font = ImageFont.load_default()

        for i, pos in st.session_state.lms.items():
            is_active = (i == target_idx)
            color = "#00FF00" if i < 15 else "#FF00FF"
            r = int(ow * 0.007)
            if is_active:
                draw.ellipse([pos[0]-r-10, pos[1]-r-10, pos[0]+r+10, pos[1]+r+10], outline="red", width=10)
            draw.ellipse([pos[0]-r, pos[1]-r, pos[0]+r, pos[1]+r], fill=color)

        res = streamlit_image_coordinates(img_raw, width=ui_width, key="canvas")
        if res:
            scale = ow / ui_width
            new_p = [int(res["x"] * scale), int(res["y"] * scale)]
            if st.session_state.lms[target_idx] != new_p:
                st.session_state.lms[target_idx] = new_p
                st.rerun()

    with col_detail:
        st.subheader("🔍 Zoom")
        cur_pos = st.session_state.lms[target_idx]
        z = 120
        box = (max(0, cur_pos[0]-z), max(0, cur_pos[1]-z), min(ow, cur_pos[0]+z), min(oh, cur_pos[1]+z))
        crop = img_raw.crop(box)
        st.image(crop, use_container_width=True)
        
        # Nudge buttons
        c1, c2, c3 = st.columns(3)
        if c2.button("🔼"): st.session_state.lms[target_idx][1] -= 1; st.rerun()
        k1, k2, k3 = st.columns(3)
        if k1.button("◀️"): st.session_state.lms[target_idx][0] -= 1; st.rerun()
        if k3.button("▶️"): st.session_state.lms[target_idx][0] += 1; st.rerun()
        if k2.button("🔽"): st.session_state.lms[target_idx][1] += 1; st.rerun()
else:
    st.info("Please upload a file.")
