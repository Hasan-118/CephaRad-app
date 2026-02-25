import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import gdown
import gc
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
import plotly.express as px_chart
from arabic_reshaper import reshape
from bidi.algorithm import get_display

# --- ۱. تنظیمات اولیه و متون راست‌چین ---
st.set_page_config(page_title="Aariz Precision Station V8.2.5", layout="wide")

def fix_text(t):
    return get_display(reshape(str(t)))

# --- ۲. معماری مرجع (بدون تغییر در ساختار لایه‌ها) ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_prob=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True), nn.Dropout2d(p=dropout_prob),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class CephaUNet(nn.Module):
    def __init__(self, n_landmarks=29):
        super().__init__()
        self.inc = DoubleConv(1, 64); self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512, dropout_prob=0.3))
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2); self.conv_up1 = DoubleConv(512, 256, dropout_prob=0.3)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.conv_up2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2); self.conv_up3 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_landmarks, kernel_size=1)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3)
        x = self.up1(x4); x = torch.cat([x, x3], dim=1); x = self.conv_up1(x)
        x = self.up2(x); x = torch.cat([x, x2], dim=1); x = self.conv_up2(x)
        x = self.up3(x); x = torch.cat([x, x1], dim=1); x = self.conv_up3(x)
        return self.outc(x)

# --- ۳. بارگذاری هوشمند مدل‌ها ---
@st.cache_resource
def load_aariz_models():
    # لینک‌های مستقیم گوگل درایو طبق دیتابیس شما
    model_ids = {
        'checkpoint_unet_clinical.pth': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks', 
        'specialist_pure_model.pth': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU', 
        'tmj_specialist_model.pth': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_models = []
    for f, fid in model_ids.items():
        if not os.path.exists(f): 
            gdown.download(f'https://drive.google.com/uc?id={fid}', f, quiet=True)
        try:
            m = CephaUNet(n_landmarks=29).to(device)
            ckpt = torch.load(f, map_location=device)
            state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            m.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=False)
            m.eval(); loaded_models.append(m)
        except: pass
    return loaded_models, device

def run_precise_prediction(img_pil, models, device):
    ow, oh = img_pil.size; img_gray = img_pil.convert('L'); ratio = 512 / max(ow, oh)
    nw, nh = int(ow * ratio), int(oh * ratio); img_rs = img_gray.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("L", (512, 512)); px, py = (512 - nw) // 2, (512 - nh) // 2
    canvas.paste(img_rs, (px, py)); input_tensor = transforms.ToTensor()(canvas).unsqueeze(0).to(device)
    
    with torch.no_grad(): 
        outs = [m(input_tensor)[0].cpu().numpy() for m in models]
    
    # تفکیک نواحی تخصصی طبق متدولوژی Aariz
    ANT_IDX = [10, 14, 9, 5, 28, 20] # Anterior
    POST_IDX = [7, 11, 12, 15]      # Posterior/TMJ
    
    coords = {}
    for i in range(29):
        # انتخاب مدل برتر برای هر نقطه
        src = outs[1][i] if i in ANT_IDX else (outs[2][i] if i in POST_IDX else outs[0][i])
        idx = np.unravel_index(np.argmax(src), (512, 512))
        coords[i] = [int((idx[1] - px) / ratio), int((idx[0] - py) / ratio)]
    
    gc.collect(); return coords

# --- ۴. رابط کاربری Streamlit ---
models, device = load_aariz_models()
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

st.sidebar.title(fix_text("ایستگاه دقیق عریض"))
pixel_size = st.sidebar.number_input("Pixel Size (mm):", 0.01, 1.0, 0.1, step=0.001, format="%.4f")
uploaded_file = st.sidebar.file_uploader(fix_text("انتخاب تصویر X-Ray:"), type=['png', 'jpg', 'jpeg'])

if uploaded_file and len(models) == 3:
    raw_img = Image.open(uploaded_file).convert("RGB"); W, H = raw_img.size
    
    if "lms" not in st.session_state or st.session_state.get("last_file") != uploaded_file.name:
        st.session_state.lms = run_precise_prediction(raw_img, models, device)
        st.session_state.last_file = uploaded_file.name
        st.session_state.click_ver = 0

    target_idx = st.sidebar.selectbox(fix_text("ویرایش نقطه:"), range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")
    
    # نمایش تصویر و سیستم کلیک
    draw_img = raw_img.copy(); draw = ImageDraw.Draw(draw_img)
    for i, p in st.session_state.lms.items():
        color = "red" if i == target_idx else "green"
        draw.ellipse([p[0]-8, p[1]-8, p[0]+8, p[1]+8], fill=color, outline="white")

    st.subheader(fix_text("🖼 آنالیز سفالومتری هوشمند"))
    res = streamlit_image_coordinates(draw_img, width=1000, key=f"main_{st.session_state.click_ver}")
    
    if res:
        scale = W / 1000
        new_coord = [int(res["x"] * scale), int(res["y"] * scale)]
        if st.session_state.lms[target_idx] != new_coord:
            st.session_state.lms[target_idx] = new_coord
            st.session_state.click_ver += 1
            st.rerun()

    # --- ۵. محاسبات آنالیز ---
    st.divider()
    l = st.session_state.lms
    def get_angle(p1, p2, p3):
        v1, v2 = np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)
        ang = np.degrees(np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))))
        return round(ang, 2)

    sna = get_angle(l[10], l[4], l[0])
    snb = get_angle(l[10], l[4], l[2])
    anb = round(sna - snb, 2)

    c1, c2 = st.columns(2)
    c1.metric("SNA Angle", f"{sna}°")
    c2.metric("ANB Angle", f"{anb}°", delta="Class II" if anb > 4 else "Class III" if anb < 0 else "Normal")

    # خروجی گرافیکی Steiner
    df = pd.DataFrame({'Metric': ['SNA', 'SNB', 'ANB'], 'Patient': [sna, snb, anb], 'Norm': [82, 80, 2]})
    st.plotly_chart(px_chart.bar(df, x='Metric', y=['Patient', 'Norm'], barmode='group'), width='stretch')
