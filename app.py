import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import gdown
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۱. ساختار مدل مرجع (ثابت) ---
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

# --- ۲. توابع کمکی ---
@st.cache_resource
def load_aariz_system():
    model_ids = {
        'checkpoint_unet_clinical.pth': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks',
        'specialist_pure_model.pth': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU',
        'tmj_specialist_model.pth': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'
    }
    device = torch.device("cpu")
    loaded_models = []
    for f, fid in model_ids.items():
        if not os.path.exists(f):
            gdown.download(f'https://drive.google.com/uc?id={fid}', f, quiet=True)
        try:
            m = CephaUNet(n_landmarks=29).to(device)
            ckpt = torch.load(f, map_location=device)
            state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            m.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=False)
            m.eval()
            loaded_models.append(m)
        except: pass
    return loaded_models, device

def get_magnified_crop(img, coord, zoom_factor=4, crop_size=100):
    x, y = coord
    left, top = max(0, x - crop_size//2), max(0, y - crop_size//2)
    right, bottom = min(img.width, x + crop_size//2), min(img.height, y + crop_size//2)
    crop = img.crop((left, top, right, bottom))
    return crop.resize((crop.width * zoom_factor, crop.height * zoom_factor), Image.NEAREST)

# --- ۳. رابط کاربری اصلی ---
st.set_page_config(page_title="Aariz AI Station V2.7", layout="wide")
models, device = load_aariz_system()
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

uploaded_file = st.sidebar.file_uploader("آپلود تصویر سفالومتری:", type=['png', 'jpg', 'jpeg'])

if uploaded_file and models:
    raw_img = Image.open(uploaded_file).convert("RGB")
    
    if "lms" not in st.session_state or st.session_state.get("file_id") != uploaded_file.name:
        # AI Logic... (Ensemble Prediction)
        img_gray = raw_img.convert('L').resize((512, 512), Image.LANCZOS)
        t = transforms.ToTensor()(img_gray).unsqueeze(0).to(device)
        with torch.no_grad():
            outs = [m(t)[0].cpu().numpy() for m in models]
        coords = {}
        sx, sy = raw_img.width/512, raw_img.height/512
        ANT_IDX, POST_IDX = [10, 14, 9, 5, 28, 20], [7, 11, 12, 15]
        for i in range(29):
            hm = outs[1][i] if i in ANT_IDX else (outs[2][i] if i in POST_IDX else outs[0][i])
            y, x = np.unravel_index(np.argmax(hm), hm.shape)
            coords[i] = [int(x * sx), int(y * sy)]
        st.session_state.lms = coords
        st.session_state.file_id = uploaded_file.name

    target_idx = st.sidebar.selectbox("نقطه برای اصلاح کلیک نهایی:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")

    col1, col2 = st.columns([2.5, 1])
    
    with col1:
        # مکان موقت برای ذخیره مختصات موس
        if "mouse_pos" not in st.session_state:
            st.session_state.mouse_pos = st.session_state.lms[target_idx]

        # نمایش ذره‌بین زنده
        st.write("🔍 **Live View (ناحیه زیر موس):**")
        mag = get_magnified_crop(raw_img, st.session_state.mouse_pos)
        st.image(mag, width=280)

        # آماده‌سازی تصویر برای نمایش
        draw_img = raw_img.copy()
        draw = ImageDraw.Draw(draw_img)
        l = st.session_state.lms
        for i, pos in l.items():
            color = "red" if i == target_idx else "lime"
            r = 14 if i == target_idx else 7
            draw.ellipse([pos[0]-r, pos[1]-r, pos[0]+r, pos[1]+r], fill=color, outline="white")

        # استفاده از on_move برای ذره‌بین زنده
        value = streamlit_image_coordinates(draw_img, width=800, key="aariz_live", on_move=True)

        if value:
            scale = raw_img.width / 800
            current_mouse = [int(value["x"]*scale), int(value["y"]*scale)]
            
            # به‌روزرسانی ذره‌بین در هر حرکت موس
            if st.session_state.mouse_pos != current_mouse:
                st.session_state.mouse_pos = current_mouse
                # اگر کلیک هم کرده باشد (MouseDown)
                if value.get("mousedown", False):
                    st.session_state.lms[target_idx] = current_mouse
                st.rerun()

    with col2:
        st.header("📊 Analysis")
        # [بخش گزارش عددی SNA/SNB مشابه قبل...]
        def get_a(p1, p2, p3):
            v1, v2 = np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)
            return round(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1, 1))), 1)
        sna, snb = get_a(l[10], l[4], l[0]), get_a(l[10], l[4], l[2])
        st.metric("SNA", f"{sna}°")
        st.metric("SNB", f"{snb}°")
        st.metric("ANB", f"{round(sna - snb, 1)}°")
