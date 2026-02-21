import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import gdown
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- تنظیمات اولیه ---
@st.cache_resource
def download_models():
    model_ids = {
        'checkpoint_unet_clinical.pth': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks',
        'specialist_pure_model.pth': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU',
        'tmj_specialist_model.pth': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'
    }
    for filename, fid in model_ids.items():
        if not os.path.exists(filename):
            url = f'https://drive.google.com/uc?id={fid}'
            gdown.download(url, filename, quiet=False)

# --- معماری مدل ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
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
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
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

@st.cache_resource
def load_aariz_models():
    download_models()
    files = ['checkpoint_unet_clinical.pth', 'specialist_pure_model.pth', 'tmj_specialist_model.pth']
    device = torch.device("cpu")
    models = []
    for f in files:
        if os.path.exists(f):
            m = CephaUNet(n_landmarks=29).to(device)
            ckpt = torch.load(f, map_location=device)
            state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            m.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=False)
            m.eval()
            models.append(m)
    return models, device

def run_ai_prediction(img_pil, models, device):
    ow, oh = img_pil.size
    img_gray = img_pil.convert('L')
    ratio = 512 / max(ow, oh)
    nw, nh = int(ow * ratio), int(oh * ratio)
    img_rs = img_gray.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("L", (512, 512))
    px, py = (512 - nw) // 2, (512 - nh) // 2
    canvas.paste(img_rs, (px, py))
    input_t = transforms.ToTensor()(canvas).unsqueeze(0).to(device)
    with torch.no_grad():
        outs = [m(input_t)[0].cpu().float().numpy() for m in models]
    coords = {}
    for i in range(29):
        hm = outs[0][i] # استفاده از مدل اصلی برای تست
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        coords[i] = [int((x - px) / ratio), int((y - py) / ratio)]
    return coords

# --- رابط کاربری ---
st.set_page_config(page_title="Aariz Station", layout="wide")
models, device = load_aariz_models()
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

uploaded_file = st.sidebar.file_uploader("انتخاب تصویر:", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    img_raw = Image.open(uploaded_file).convert("RGB")
    
    # ریست کردن مختصات اگر فایل جدید آپلود شد
    if "fid" not in st.session_state or st.session_state.fid != uploaded_file.name:
        st.session_state.lms = run_ai_prediction(img_raw, models, device)
        st.session_state.fid = uploaded_file.name

    target_idx = st.sidebar.selectbox("نقطه برای اصلاح:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")

    col1, col2 = st.columns([3, 1])
    with col1:
        # رسم نقاط روی یک کپی از تصویر
        draw_img = img_raw.copy()
        draw = ImageDraw.Draw(draw_img)
        for i, pos in st.session_state.lms.items():
            r = 12 if i == target_idx else 6
            color = "red" if i == target_idx else "lime"
            draw.ellipse([pos[0]-r, pos[1]-r, pos[0]+r, pos[1]+r], fill=color, outline="white")

        # نمایش با عرض ثابت برای جلوگیری از خطای مختصات
        st.write("### برای جابجایی نقطه روی تصویر کلیک کنید")
        value = streamlit_image_coordinates(draw_img, width=800, key="main_canvas")
        
        if value:
            scale = img_raw.width / 800
            new_x, new_y = int(value["x"] * scale), int(value["y"] * scale)
            if st.session_state.lms[target_idx] != [new_x, new_y]:
                st.session_state.lms[target_idx] = [new_x, new_y]
                st.rerun()
else:
    st.warning("لطفاً یک تصویر آپلود کنید.")
