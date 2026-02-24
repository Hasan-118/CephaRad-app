import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import gdown
import gc
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۱. معماری مرجع Aariz (بدون دستکاری) ---
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

# --- ۲. بارگذاری هوشمند مدل‌ها ---
@st.cache_resource
def load_aariz_models():
    model_ids = {
        'checkpoint_unet_clinical.pth': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks', 
        'specialist_pure_model.pth': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU', 
        'tmj_specialist_model.pth': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'
    }
    device = torch.device("cpu"); loaded_models = []
    for f, fid in model_ids.items():
        if not os.path.exists(f): gdown.download(f'https://drive.google.com/uc?id={fid}', f, quiet=True)
        try:
            m = CephaUNet(n_landmarks=29).to(device); ckpt = torch.load(f, map_location=device)
            state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            m.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=False)
            m.eval(); loaded_models.append(m)
        except: pass
    return loaded_models, device

# --- ۳. منطق پیش‌بینی دقیق (تفکیک نواحی تخصصی) ---
def run_precise_prediction(img_pil, models, device):
    ow, oh = img_pil.size; img_gray = img_pil.convert('L'); ratio = 512 / max(ow, oh)
    nw, nh = int(ow * ratio), int(oh * ratio); img_rs = img_gray.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("L", (512, 512)); px, py = (512 - nw) // 2, (512 - nh) // 2
    canvas.paste(img_rs, (px, py)); input_tensor = transforms.ToTensor()(canvas).unsqueeze(0).to(device)
    
    with torch.no_grad(): outs = [m(input_tensor)[0].cpu().numpy() for m in models]
    
    # اندیس‌های تخصصی (بر اساس مرجع V7.8)
    ANT_IDX, POST_IDX = [10, 14, 9, 5, 28, 20], [7, 11, 12, 15]
    
    coords = {}
    for i in range(29):
        # انتخاب مدل بر اساس ناحیه
        m_idx = 1 if i in ANT_IDX else (2 if i in POST_IDX else 0)
        heatmap = outs[m_idx][i]
        y, x = np.unravel_index(np.argmax(heatmap), (512, 512))
        coords[i] = [int((x - px) / ratio), int((y - py) / ratio)]
    
    gc.collect(); return coords

# --- ۴. توابع محاسباتی (اصلاح شده برای NumPy 2.0 مطابق لاگ) ---
def get_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2); v2 = np.array(p3) - np.array(p2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return round(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (norm if norm > 0 else 1e-6), -1, 1))), 2)

def dist_to_line(p, l1, l2):
    # اصلاح لاگ NumPy 2.0: تبدیل به بردار ۳ بعدی برای np.cross
    p3d, l1_3d, l2_3d = np.append(p, 0), np.append(l1, 0), np.append(l2, 0)
    return np.linalg.norm(np.cross(l2_3d-l1_3d, l1_3d-p3d)) / (np.linalg.norm(l2_3d-l1_3d) + 1e-6)

# --- ۵. رابط کاربری Streamlit Cloud ---
st.set_page_config(page_title="Aariz Precision Station V7.8", layout="wide")
models, device = load_aariz_models()
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

if "click_version" not in st.session_state: st.session_state.click_version = 0

st.sidebar.title("🛠 Aariz Console")
pixel_size = st.sidebar.number_input("Pixel Size (mm):", 0.01, 1.0, 0.1, format="%.4f")
uploaded_file = st.sidebar.file_uploader("Cephalometric Image:", type=['png', 'jpg', 'jpeg'])

if uploaded_file and len(models) == 3:
    raw_img = Image.open(uploaded_file).convert("RGB"); W, H = raw_img.size
    
    if "lms" not in st.session_state or st.session_state.get("file_id") != uploaded_file.name:
        st.session_state.lms = run_precise_prediction(raw_img, models, device)
        st.session_state.file_id = uploaded_file.name

    target_idx = st.sidebar.selectbox("🎯 Active Landmark:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")

    col1, col2 = st.columns([1.2, 2.5])
    
    # بخش Micro-Adjustment (سمت چپ)
    with col1:
        st.subheader("🔍 Micro-Adjustment")
        l_pos = st.session_state.lms[target_idx]; size_m = 160
        left, top = max(0, min(int(l_pos[0]-size_m//2), W-size_m)), max(0, min(int(l_pos[1]-size_m//2), H-size_m))
        mag_crop = raw_img.crop((left, top, left+size_m, top+size_m)).resize((400, 400), Image.LANCZOS)
        
        # نشانگر مرکزی
        mag_draw = ImageDraw.Draw(mag_crop)
        mag_draw.line((190, 200, 210, 200), fill="red", width=2); mag_draw.line((200, 190, 200, 210), fill="red", width=2)
        
        res_mag = streamlit_image_coordinates(mag_crop, key=f"mag_{st.session_state.click_version}")
        if res_mag:
            scale = size_m / 400
            new_x, new_y = int(left + (res_mag["x"] * scale)), int(top + (res_mag["y"] * scale))
            if [new_x, new_y] != st.session_state.lms[target_idx]:
                st.session_state.lms[target_idx] = [new_x, new_y]
                st.session_state.click_version += 1; st.rerun()

    # بخش نمایش اصلی (سمت راست)
    with col2:
        st.subheader("🖼 Cephalometric Analysis")
        draw_img = raw_img.copy(); draw = ImageDraw.Draw(draw_img); l = st.session_state.lms
        
        # ترسیم خطوط آنالیز پایه
        if all(k in l for k in [10, 4, 0, 2]): # S, N, A, B
            draw.line([tuple(l[10]), tuple(l[4])], fill="yellow", width=3) # S-N
            draw.line([tuple(l[4]), tuple(l[0])], fill="cyan", width=2)   # N-A
            draw.line([tuple(l[4]), tuple(l[2])], fill="magenta", width=2)# N-B
            
        # رسم نقاط
        for i, pos in l.items():
            color = (255, 0, 0) if i == target_idx else (0, 255, 0)
            draw.ellipse([pos[0]-8, pos[1]-8, pos[0]+8, pos[1]+8], fill=color, outline="white")
            draw.text((pos[0]+12, pos[1]-12), landmark_names[i], fill=color)

        streamlit_image_coordinates(draw_img, width=800)

    # --- ۶. گزارش بالینی نهایی ---
    st.divider()
    sna = get_angle(l[10], l[4], l[0]); snb = get_angle(l[10], l[4], l[2]); anb = round(sna - snb, 2)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("SNA", f"{sna}°")
    c2.metric("SNB", f"{snb}°")
    c3.metric("ANB", f"{anb}°")
    
    st.info(f"**تفسیر بالینی اسکلتال:** {'Class II' if anb > 4 else 'Class III' if anb < 0 else 'Class I'}")
