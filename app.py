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

# --- ۱. معماری شبکه عصبی (بدون تغییر در ساختار اصلی) ---
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

# --- ۲. مدیریت بارگذاری هوشمند مدل‌ها ---
@st.cache_resource
def load_aariz_models():
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

def run_precise_prediction(img_pil, models, device):
    ow, oh = img_pil.size
    img_gray = img_pil.convert('L')
    ratio = 512 / max(ow, oh)
    nw, nh = int(ow * ratio), int(oh * ratio)
    img_rs = img_gray.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("L", (512, 512))
    px, py = (512 - nw) // 2, (512 - nh) // 2
    canvas.paste(img_rs, (px, py))
    input_tensor = transforms.ToTensor()(canvas).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outs = [m(input_tensor)[0].cpu().numpy() for m in models]
    
    # تفکیک نواحی تخصصی بر اساس تجربه مدل‌ها
    ANT_IDX = [10, 14, 9, 5, 28, 20] # S, Go, R, Or, Sn, UIA
    POST_IDX = [7, 11, 12, 15] # PNS, Ar, Co, Po
    
    coords = {}
    for i in range(29):
        hm = outs[1][i] if i in ANT_IDX else (outs[2][i] if i in POST_IDX else outs[0][i])
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        coords[i] = [int((x - px) / ratio), int((y - py) / ratio)]
    
    gc.collect()
    return coords

# --- ۳. تنظیمات و رابط کاربری ---
st.set_page_config(page_title="Aariz Precision V7.8", layout="wide")
models, device = load_aariz_models()
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

if "v" not in st.session_state: st.session_state.v = 0

st.sidebar.title("🩺 Control Panel")
gender = st.sidebar.selectbox("Patient Gender:", ["Male", "Female"])
px_size = st.sidebar.number_input("Pixel Size (mm):", 0.01, 0.5, 0.1, format="%.4f")

uploaded_file = st.sidebar.file_uploader("Upload Cephalogram:", type=['png', 'jpg', 'jpeg'])

if uploaded_file and len(models) == 3:
    raw = Image.open(uploaded_file).convert("RGB")
    W, H = raw.size
    
    if "lms" not in st.session_state or st.session_state.get("fid") != uploaded_file.name:
        st.session_state.lms = run_precise_prediction(raw, models, device)
        st.session_state.fid = uploaded_file.name

    target_idx = st.sidebar.selectbox("🎯 Select Landmark:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")
    lms = st.session_state.lms

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔍 Precision Zoom")
        p = lms[target_idx]
        box = 80
        # اصلاح امنیتی کراپ: جلوگیری از خروج از کادر
        left = max(0, min(p[0] - box, W - 2*box))
        top = max(0, min(p[1] - box, H - 2*box))
        crop = raw.crop((left, top, left + 2*box, top + 2*box)).resize((350, 350), Image.LANCZOS)
        
        # رسم نشانه مرکزی در ذره‌بین
        c_draw = ImageDraw.Draw(crop)
        c_draw.line((165, 175, 185, 175), fill="red", width=2)
        c_draw.line((175, 165, 175, 185), fill="red", width=2)
        
        res_m = streamlit_image_coordinates(crop, key=f"m_{target_idx}_{st.session_state.v}")
        if res_m:
            scale = (2 * box) / 350
            new_x = int(left + (res_m["x"] * scale))
            new_y = int(top + (res_m["y"] * scale))
            if lms[target_idx] != [new_x, new_y]:
                lms[target_idx] = [new_x, new_y]
                st.session_state.v += 1
                st.rerun()

    with col2:
        st.subheader("📊 Cephalometric Tracing")
        disp = raw.copy(); draw = ImageDraw.Draw(disp)
        
        # رسم خطوط آنالیز بالینی (S-N و FH)
        if all(k in lms for k in [10, 4, 15, 5]):
            draw.line([tuple(lms[10]), tuple(lms[4])], fill="yellow", width=4) # S-N
            draw.line([tuple(lms[15]), tuple(lms[5])], fill="purple", width=4) # FH Plane
        
        for i, pt in lms.items():
            color = (255, 0, 0) if i == target_idx else (0, 255, 0)
            radius = 12 if i == target_idx else 7
            draw.ellipse([pt[0]-radius, pt[1]-radius, pt[0]+radius, pt[1]+radius], fill=color, outline="white", width=2)
        
        res_main = streamlit_image_coordinates(disp, width=800, key=f"main_{st.session_state.v}")
        if res_main:
            sc_w = W / 800
            new_x, new_y = int(res_main["x"] * sc_w), int(res_main["y"] * sc_w)
            if lms[target_idx] != [new_x, new_y]:
                lms[target_idx] = [new_x, new_y]
                st.session_state.v += 1
                st.rerun()

    # --- ۴. محاسبات بالینی با رفع خطای Numpy 2.0 ---
    st.divider()
    def get_angle(p1, p2, p3):
        v1 = np.array(p1, dtype=np.float64) - np.array(p2, dtype=np.float64)
        v2 = np.array(p3, dtype=np.float64) - np.array(p2, dtype=np.float64)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0: return 0
        return round(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0))), 2)

    # زوایای اصلی
    sna = get_angle(lms[10], lms[4], lms[0])
    snb = get_angle(lms[10], lms[4], lms[2])
    anb = round(sna - snb, 2)

    st.header("📋 Clinical Diagnostics")
    c1, c2, c3 = st.columns(3)
    c1.metric("SNA Angle", f"{sna}°")
    c2.metric("SNB Angle", f"{snb}°")
    c3.metric("ANB (Skeletal Class)", f"{anb}°")
    
    diag = "Class II" if anb > 4 else ("Class III" if anb < 0 else "Class I")
    st.success(f"**Diagnostic Summary:** The patient exhibits a **{diag}** skeletal relationship.")
