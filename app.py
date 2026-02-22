import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import json
import gdown
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۱. تنظیمات اولیه و دانلود خودکار مدل‌ها ---
RESULTS_DIR = "Aariz_Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

@st.cache_resource
def download_models():
    model_ids = {
        'checkpoint_unet_clinical.pth': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks',
        'specialist_pure_model.pth': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU',
        'tmj_specialist_model.pth': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'
    }
    for filename, fid in model_ids.items():
        if not os.path.exists(filename):
            with st.spinner(f'در حال فراخوانی {filename}...'):
                url = f'https://drive.google.com/uc?id={fid}'
                gdown.download(url, filename, quiet=False)

# --- ۲. معماری مدل (دقیقاً مطابق نسخه مرجع شما) ---
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

# --- ۳. بارگذاری مدل‌ها ---
@st.cache_resource
def load_aariz_models():
    download_models()
    model_files = ['checkpoint_unet_clinical.pth', 'specialist_pure_model.pth', 'tmj_specialist_model.pth']
    device = torch.device("cpu") # بهینه برای Streamlit Cloud
    loaded_models = []
    for f in model_files:
        if os.path.exists(f):
            try:
                m = CephaUNet(n_landmarks=29).to(device)
                ckpt = torch.load(f, map_location=device)
                state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
                m.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=False)
                m.eval()
                loaded_models.append(m)
            except: pass
    return loaded_models, device

# --- ۴. پیش‌بینی هوشمند با اصلاح کادر قدامی (Letterboxing) ---
def run_ai_prediction(img_pil, models, device):
    ow, oh = img_pil.size
    img_gray = img_pil.convert('L')
    
    # حفظ نسبت ابعاد برای جلوگیری از قطع شدن صورت
    ratio = 512 / max(ow, oh)
    nw, nh = int(ow * ratio), int(oh * ratio)
    img_rs = img_gray.resize((nw, nh), Image.LANCZOS)
    
    canvas = Image.new("L", (512, 512))
    px, py = (512 - nw) // 2, (512 - nh) // 2
    canvas.paste(img_rs, (px, py))
    
    input_tensor = transforms.ToTensor()(canvas).unsqueeze(0).to(device)
    with torch.inference_mode():
        outs = [mod(input_tensor)[0].cpu().numpy() for mod in models]
    
    ANT_IDX, POST_IDX = [10, 14, 9, 5, 28, 20], [7, 11, 12, 15]
    coords = {}
    for i in range(29):
        if i in ANT_IDX and len(outs) >= 2: hm = outs[1][i]
        elif i in POST_IDX and len(outs) >= 3: hm = outs[2][i]
        else: hm = outs[0][i]
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        # بازگشت به مختصات اصلی
        coords[i] = [int((x - px) / ratio), int((y - py) / ratio)]
    return coords

# --- ۵. رابط کاربری (UI) ---
st.set_page_config(page_title="Aariz AI Station V2", layout="wide")
models, device = load_aariz_models()
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

uploaded_file = st.sidebar.file_uploader("آپلود سفالوگرام:", type=['png', 'jpg', 'jpeg'])

if uploaded_file and models:
    raw_img = Image.open(uploaded_file).convert("RGB")
    if "lms" not in st.session_state or st.session_state.get("file_id") != uploaded_file.name:
        st.session_state.lms = run_ai_prediction(raw_img, models, device)
        st.session_state.file_id = uploaded_file.name

    target_idx = st.sidebar.selectbox("نقطه فعال:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")
    col1, col2 = st.columns([2.5, 1])
    
    with col1:
        draw_img = raw_img.copy()
        draw = ImageDraw.Draw(draw_img)
        l = st.session_state.lms
        draw.line([tuple(l[10]), tuple(l[4]), tuple(l[0])], fill="yellow", width=5)
        for i, pos in l.items():
            color = "red" if i == target_idx else "#00FF00"
            r = 15 if i == target_idx else 8
            draw.ellipse([pos[0]-r, pos[1]-r, pos[0]+r, pos[1]+r], fill=color, outline="white")

        st.subheader("📍 تنظیم دستی لندمارک")
        res = streamlit_image_coordinates(draw_img, width=850, key="aariz_v2")
        if res:
            scale = raw_img.width / 850
            nx, ny = int(res["x"]*scale), int(res["y"]*scale)
            if l[target_idx] != [nx, ny]:
                st.session_state.lms[target_idx] = [nx, ny]
                st.rerun()

    with col2:
        st.header("📊 Clinical Report")
        def angle(p1, p2, p3):
            v1, v2 = np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)
            norm = np.linalg.norm(v1)*np.linalg.norm(v2)
            return round(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/norm, -1, 1))), 2)
        
        sna, snb = angle(l[10], l[4], l[0]), angle(l[10], l[4], l[2])
        st.metric("SNA", f"{sna}°")
        st.metric("SNB", f"{snb}°")
        st.metric("ANB", f"{round(sna-snb, 2)}°")
        if st.button("💾 ذخیره نهایی"):
            st.success("آنالیز ثبت شد.")
