import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۱. معماری فیکس شده برای لود شدن مدل‌ها ---
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

# --- ۲. لودر Ensemble ---
@st.cache_resource
def load_models():
    paths = ['checkpoint_unet_clinical.pth', 'specialist_pure_model.pth', 'tmj_specialist_model.pth']
    models = []
    for p in paths:
        if os.path.exists(p):
            m = CephaUNet(n_landmarks=29)
            ckpt = torch.load(p, map_location="cpu")
            m.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
            m.eval()
            models.append(m)
    return models

def predict_ensemble(img_path, models):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    oh, ow = img.shape
    img_res = cv2.resize(img, (384, 384))
    input_t = transforms.ToTensor()(img_res).unsqueeze(0)
    all_hms = []
    with torch.no_grad():
        for m in models: all_hms.append(m(input_t)[0].numpy())
    avg_hm = np.mean(all_hms, axis=0)
    lms = {}
    for i in range(29):
        y, x = np.unravel_index(np.argmax(avg_hm[i]), (384,384))
        lms[i] = [int(x * ow / 384), int(y * oh / 384)]
    return lms, (ow, oh)

# --- ۳. رابط کاربری ---
st.set_page_config(layout="wide", page_title="Aariz Control")
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']
models = load_models()

# سایدبار
st.sidebar.header("⚙️ Settings")
ui_width = st.sidebar.slider("Display Width (px)", 400, 1400, 800) # اسلایدر اینجاست
path_input = st.sidebar.text_input("Project Path", value=os.getcwd())
img_dir = os.path.join(path_input, "Aariz", "train", "Cephalograms")

if os.path.exists(img_dir) and models:
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))])
    selected_file = st.sidebar.selectbox("Image", files)
    target_idx = st.sidebar.selectbox("Active Point", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")

    full_path = os.path.join(img_dir, selected_file)
    if "lms" not in st.session_state or st.session_state.get("file") != selected_file:
        st.session_state.lms, st.session_state.orig_size = predict_ensemble(full_path, models)
        st.session_state.file = selected_file

    col_left, col_right = st.columns([2.5, 1])

    with col_left:
        img_pil = Image.open(full_path).convert("RGB")
        ow, oh = st.session_state.orig_size
        draw = ImageDraw.Draw(img_pil)
        
        # فونت بزرگ
        try: font = ImageFont.truetype("arialbd.ttf", int(ow * 0.035))
        except: font = ImageFont.load_default()

        for i, pos in st.session_state.lms.items():
            is_active = (i == target_idx)
            r = int(ow * 0.008)
            color = "#00FF00" if i < 15 else "#FF00FF"
            if is_active:
                draw.ellipse([pos[0]-r-10, pos[1]-r-10, pos[0]+r+10, pos[1]+r+10], outline="red", width=12)
            draw.ellipse([pos[0]-r, pos[1]-r, pos[0]+r, pos[1]+r], fill=color, outline="white", width=4)
            draw.text((pos[0]+r+10, pos[1]-r-20), f"{i}:{landmark_names[i]}", fill="yellow", font=font, stroke_width=6, stroke_fill="black")

        st.subheader("📍 Main View")
        # استفاده از ui_width که از اسلایدر می‌آید
        res = streamlit_image_coordinates(img_pil, width=ui_width, key="main_img")
        
        if res:
            scale = ow / ui_width
            st.session_state.lms[target_idx] = [int(res["x"] * scale), int(res["y"] * scale)]
            st.rerun()

    with col_right:
        st.subheader("🔍 Precision Zoom")
        # مختصات لحظه‌ای نقطه فعال برای زوم
        cur_pos = st.session_state.lms[target_idx]
        z_size = 150
        # محاسبه کادر کراپ بر اساس نقطه فعلی
        box = (max(0, cur_pos[0]-z_size), max(0, cur_pos[1]-z_size), 
               min(ow, cur_pos[0]+z_size), min(oh, cur_pos[1]+z_size))
        
        zoom_img = Image.open(full_path).convert("RGB").crop(box)
        z_draw = ImageDraw.Draw(zoom_img)
        zw, zh = zoom_img.size
        z_draw.line([(zw//2, 0), (zw//2, zh)], fill="red", width=3)
        z_draw.line([(0, zh//2), (zw, zh//2)], fill="red", width=3)
        st.image(zoom_img, use_container_width=True)
        
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        if c2.button("🔼"): st.session_state.lms[target_idx][1] -= 1; st.rerun()
        if c1.button("◀️"): st.session_state.lms[target_idx][0] -= 1; st.rerun()
        if c3.button("▶️"): st.session_state.lms[target_idx][0] += 1; st.rerun()
        if c2.button("🔽"): st.session_state.lms[target_idx][1] += 1; st.rerun()
        
    st.divider()
    
    l = st.session_state.lms
    st.write(f"Active: {landmark_names[target_idx]} at {st.session_state.lms[target_idx]}")

else:
    st.error("Missing path or models.")
