import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import gdown
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۱. ساختار مدل مرجع (بدون تغییر) ---
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
        self.inc = DoubleConv(1, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
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

# --- ۲. توابع لودر و محاسبات دقیق ---
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
            m.eval(); loaded_models.append(m)
        except: pass
    return loaded_models, device

# --- ۳. رابط کاربری با تمرکز بر دقت بالا ---
st.set_page_config(page_title="Aariz Precision V3.6", layout="wide")
models, device = load_aariz_system()
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

uploaded_file = st.sidebar.file_uploader("آپلود تصویر سفالومتری:", type=['png', 'jpg', 'jpeg'])

if uploaded_file and len(models) == 3:
    raw_img = Image.open(uploaded_file).convert("RGB")
    W, H = raw_img.size
    
    if "lms" not in st.session_state or st.session_state.get("file_id") != uploaded_file.name:
        with st.spinner("استخراج لندمارک‌ها با دقت حداکثری..."):
            # ۱. پیش‌پردازش دقیق
            img_input = raw_img.convert('L').resize((512, 512), Image.LANCZOS)
            t = transforms.ToTensor()(img_input).unsqueeze(0).to(device)
            
            # ۲. استخراج هیت‌مپ‌ها از هر ۳ مدل
            with torch.no_grad():
                preds = [m(t)[0].cpu().numpy() for m in models]
            
            coords = {}
            # فاکتور مقیاس دقیق (اعشاری)
            scale_x, scale_y = W / 512.0, H / 512.0
            
            # ۳. منطق انسمبل هوشمند (دقیقاً مشابه مرجع اول شما)
            ANT_IDX = [1, 20, 21, 22, 24, 25, 26, 28] # قدامی
            TMJ_IDX = [11, 12, 15, 16] # خلفی/کندیل
            
            for i in range(29):
                if i in ANT_IDX:
                    hm = preds[1][i] # متخصص قدامی
                elif i in TMJ_IDX:
                    hm = preds[2][i] # متخصص خلفی
                else:
                    hm = preds[0][i] # مدل جنرال
                
                y, x = np.unravel_index(np.argmax(hm), hm.shape)
                # نگاشت دقیق به مختصات تصویر اصلی
                coords[i] = [int(x * scale_x), int(y * scale_y)]
            
            st.session_state.lms = coords
            st.session_state.file_id = uploaded_file.name

    # [ادامه کد شامل ذره‌بین ایمن V3.5 و ترسیم...]
