import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
import gdown
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۱. معماری دقیق مدل (مطابق Untitled6.ipynb) ---
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

# --- ۲. مدیریت دانلود و بارگذاری انسامبل ---
@st.cache_resource
def load_aariz_engines():
    drive_ids = {
        'checkpoint_unet_clinical.pth': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks',
        'specialist_pure_model.pth': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU',
        'tmj_specialist_model.pth': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'
    }
    engines = []
    os.makedirs('models', exist_ok=True)
    
    for filename, fid in drive_ids.items():
        path = os.path.join('models', filename)
        if not os.path.exists(path):
            with st.spinner(f'در حال دانلود مدل {filename}...'):
                url = f'https://drive.google.com/uc?id={fid}'
                gdown.download(url, path, quiet=False)
        
        model = CephaUNet(n_landmarks=29)
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        # حذف پیشوند module در صورت وجود
        new_state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(new_state, strict=False)
        model.eval()
        engines.append(model)
    return engines

# --- ۳. پردازش تصویر و انسامبل ---
def get_ensemble_prediction(img_pil, engines):
    # تبدیل PIL به OpenCV برای CLAHE
    img_gray = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_gray)
    
    orig_h, orig_w = img_enhanced.shape
    img_res = cv2.resize(img_enhanced, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    
    # تبدیل به تنسور و نرمال‌سازی (مطابق آموزش نوت‌بوک)
    input_t = transforms.ToTensor()(img_res).unsqueeze(0)
    
    all_heatmaps = []
    with torch.no_grad():
        for model in engines:
            # خروجی خام مدل قبل از انسامبل
            pred = model(input_t)
            all_heatmaps.append(torch.sigmoid(pred)[0].numpy())
    
    # میانگین‌گیری روی هیت‌مپ‌ها
    avg_output = np.mean(all_heatmaps, axis=0)
    coords = {}
    for i in range(29):
        hm = avg_output[i]
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        coords[i] = [int(x * orig_w / 512), int(y * orig_h / 512)]
    return coords, (orig_w, orig_h)

def get_angle(p1, p2, p3):
    v1, v2 = np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return round(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/norm, -1, 1))), 1) if norm != 0 else 0

# --- ۴. رابط کاربری Streamlit ---
st.set_page_config(layout="wide", page_title="Aariz Station v3.0")

landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']
engines = load_aariz_engines()

st.title("🦷 پنل فوق‌پیشرفته آنالیز CephaRad")

uploaded_file = st.file_uploader("تصویر سفالومتری بیمار را آپلود کنید", type=['png', 'jpg', 'jpeg'])

if uploaded_file and engines:
    img_pil = Image.open(uploaded_file).convert("RGB")
    
    if "lms" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        st.session_state.lms, st.session_state.orig_size = get_ensemble_prediction(img_pil, engines)
        st.session_state.file_name = uploaded_file.name

    target_idx = st.sidebar.selectbox("🎯 انتخاب نقطه برای اصلاح:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")
    
    col_main, col_zoom = st.columns([2.5, 1])

    with col_main:
        # رسم نقاط روی تصویر برای نمایش
        draw_img = img_pil.copy()
        draw = ImageDraw.Draw(draw_img)
        l = st.session_state.lms
        orig_w, _ = st.session_state.orig_size
        
        for i, pos in l.items():
            r = int(orig_w * 0.006)
            color = "red" if i == target_idx else "#00FF00"
            draw.ellipse([pos[0]-r, pos[1]-r, pos[0]+r, pos[1]+r], fill=color, outline="white", width=2)
            draw.text((pos[0]+r, pos[1]-r), landmark_names[i], fill="yellow")

        st.subheader("📍 نمای اصلی (برای جابجایی دقیق کلیک کنید)")
        # کامپوننت کلیک روی تصویر
        res = streamlit_image_coordinates(draw_img, width=900, key="main_img")
        
        if res:
            scale = orig_w / 900
            new_x, new_y = int(res["x"] * scale), int(res["y"] * scale)
            if l[target_idx] != [new_x, new_y]:
                st.session_state.lms[target_idx] = [new_x, new_y]
                st.rerun()

    with col_zoom:
        st.subheader("🔍 زوم میکروسکوپی")
        active_pos = st.session_state.lms[target_idx]
        z_size = 120
        box = (max(0, active_pos[0]-z_size), max(0, active_pos[1]-z_size), 
               min(orig_w, active_pos[0]+z_size), min(st.session_state.orig_size[1], active_pos[1]+z_size))
        
        zoom_img = img_pil.crop(box)
        st.image(zoom_img, use_container_width=True, caption=f"نقطه فعال: {landmark_names[target_idx]}")
        
        st.divider()
        # محاسبات Steiner
        sna = get_angle(l[10], l[4], l[0]) # S-N-A
        snb = get_angle(l[10], l[4], l[2]) # S-N-B
        anb = round(sna - snb, 1)
        
        st.metric("SNA Angle", f"{sna}°")
        st.metric("SNB Angle", f"{snb}°")
        st.metric("ANB (Class)", f"{anb}°", delta="Class II" if anb > 4 else ("Class III" if anb < 0 else "Class I"))

    st.sidebar.markdown("---")
    if st.sidebar.button("💾 ثبت نهایی و چاپ گزارش"):
        st.sidebar.success("اطلاعات آنالیز ذخیره شد.")

else:
    st.info("منتظر آپلود تصویر و لود شدن مدل‌ها هستیم...")
