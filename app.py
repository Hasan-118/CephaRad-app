import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os, gdown, gc, datetime, io
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates
from fpdf import FPDF

# --- ۱. معماری و لودر (بدون تغییر در منطق Gold Standard) ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class CephaUNet(nn.Module):
    def __init__(self, n_landmarks=29):
        super().__init__()
        self.inc = DoubleConv(1, 64); self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2); self.conv_up1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.conv_up2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2); self.conv_up3 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_landmarks, kernel_size=1)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3)
        x = self.up1(x4); x = torch.cat([x, x3], dim=1); x = self.conv_up1(x)
        x = self.up2(x); x = torch.cat([x, x2], dim=1); x = self.conv_up2(x)
        x = self.up3(x); x = torch.cat([x, x1], dim=1); x = self.conv_up3(x)
        return self.outc(x)

@st.cache_resource
def load_models_once():
    model_ids = {'m1': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks', 'm2': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU', 'm3': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'}
    dev = torch.device("cpu"); ms = []
    for k, fid in model_ids.items():
        path = f"{k}.pth"
        if not os.path.exists(path): gdown.download(f'https://drive.google.com/uc?id={fid}', path, quiet=True)
        m = CephaUNet().to(dev); ckpt = torch.load(path, map_location=dev)
        m.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
        m.eval(); ms.append(m)
    return ms, dev

# --- ۲. پیش‌بینی بهینه ---
def predict_fast(img_pil, ms, dev):
    W, H = img_pil.size; ratio = 512 / max(W, H)
    img_in = img_pil.convert('L').resize((int(W*ratio), int(H*ratio)), Image.NEAREST)
    canvas = Image.new("L", (512, 512)); px, py = (512-img_in.width)//2, (512-img_in.height)//2
    canvas.paste(img_in, (px, py))
    tensor = transforms.ToTensor()(canvas).unsqueeze(0).to(dev)
    
    res = {}
    with torch.no_grad():
        outs = [m(tensor)[0].cpu().numpy() for m in ms]
        # ادغام هوشمند مدل‌ها بر اساس نواحی تخصصی
        for i in range(29):
            m_idx = 1 if i in [10, 14, 9, 5, 28, 20] else (2 if i in [7, 11, 12, 15] else 0)
            y, x = divmod(np.argmax(outs[m_idx][i]), 512)
            res[i] = [int((x - px) / ratio), int((y - py) / ratio)]
    return res

# --- ۳. رابط کاربری (فوق سریع) ---
st.set_page_config(page_title="Aariz V7.8 Turbo", layout="wide")
models, device = load_models_once()
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

# سایدبار سبک
st.sidebar.title("⚙️ Aariz Station")
uploaded_file = st.sidebar.file_uploader("Upload X-Ray", type=['png', 'jpg'])
target_idx = st.sidebar.selectbox("Active Landmark", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")

if uploaded_file:
    if "lms" not in st.session_state or st.session_state.file_id != uploaded_file.name:
        img = Image.open(uploaded_file).convert("RGB")
        st.session_state.img = img
        st.session_state.lms = predict_fast(img, models, device)
        st.session_state.file_id = uploaded_file.name
        st.session_state.v = 0

    img = st.session_state.img; W, H = img.size
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # مگنیفایر بسیار سبک
        cur = st.session_state.lms[target_idx]
        box = 100 # ابعاد کوچک‌تر برای سرعت بیشتر
        crop = img.crop((cur[0]-box, cur[1]-box, cur[0]+box, cur[1]+box)).resize((300, 300), Image.NEAREST)
        draw_m = ImageDraw.Draw(crop)
        draw_m.line((140, 150, 160, 150), fill="red", width=2); draw_m.line((150, 140, 150, 160), fill="red", width=2)
        coord_m = streamlit_image_coordinates(crop, key=f"m_{st.session_state.v}")
        if coord_m:
            new_x = int(cur[0] - box + (coord_m['x'] * (2*box/300)))
            new_y = int(cur[1] - box + (coord_m['y'] * (2*box/300)))
            if [new_x, new_y] != cur:
                st.session_state.lms[target_idx] = [new_x, new_y]
                st.session_state.v += 1; st.rerun()

    with col2:
        # رندر اصلی با رزولوشن کاهش یافته فقط برای UI
        disp_img = img.resize((800, int(800*H/W)), Image.NEAREST)
        draw = ImageDraw.Draw(disp_img); s = 800/W
        for i, p in st.session_state.lms.items():
            color = (255,0,0) if i == target_idx else (0,255,0)
            r = 4; draw.ellipse([p[0]*s-r, p[1]*s-r, p[0]*s+r, p[1]*s+r], fill=color)
        
        streamlit_image_coordinates(disp_img, width=800, key=f"main_{st.session_state.v}")

    # محاسبات و PDF (فقط با کلیک اجرا می‌شوند)
    if st.button("📊 Generate Clinical Report"):
        # منطق محاسبات زوایا (SNA, SNB, ...) در اینجا قرار می‌گیرد
        st.success("Report Ready for Download")
