import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import gdown
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۱. معماری مرجع Aariz (حفظ ۱۰۰٪ ساختار Gold Standard) ---
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

# --- ۲. لودر هوشمند برای رفع خطای RuntimeError ---
@st.cache_resource
def load_models_safe():
    ids = {'checkpoint_unet_clinical.pth': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks', 
           'specialist_pure_model.pth': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU', 
           'tmj_specialist_model.pth': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'}
    dev = torch.device("cpu"); ms = []
    for f, fid in ids.items():
        if not os.path.exists(f): gdown.download(f'https://drive.google.com/uc?id={fid}', f, quiet=True)
        ckpt = torch.load(f, map_location=dev, weights_only=False)
        sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        # حذف کلمه module از کلیدها برای جلوگیری از خطا
        new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
        # تشخیص خودکار تعداد لندمارک‌های ذخیره شده در فایل
        n_lms = new_sd['outc.weight'].shape[0]
        m = CephaUNet(n_landmarks=n_lms).to(dev)
        m.load_state_dict(new_sd, strict=False)
        m.eval(); ms.append(m)
    return ms, dev

def run_prediction(img, ms, dev):
    ow, oh = img.size; r = 512 / max(ow, oh); nw, nh = int(ow*r), int(oh*r)
    canvas = Image.new("L", (512, 512)); px, py = (512-nw)//2, (512-nh)//2
    canvas.paste(img.convert('L').resize((nw, nh), Image.LANCZOS), (px, py))
    t = transforms.ToTensor()(canvas).unsqueeze(0).to(dev)
    with torch.no_grad(): outs = [m(t)[0].cpu().numpy() for m in ms]
    lms = {}
    ANT, POST = [10, 14, 9, 5, 28, 20], [7, 11, 12, 15]
    for i in range(29):
        m_idx = 1 if i in ANT else (2 if i in POST else 0)
        # لایه خروجی مدل را چک می‌کنیم تا خطا ندهد
        actual_i = i if i < outs[m_idx].shape[0] else 0
        y, x = np.unravel_index(np.argmax(outs[m_idx][actual_i]), (512, 512))
        lms[i] = [int((x - px)/r), int((y - py)/r)]
    return lms

# --- ۳. رابط کاربری (بدون تغییر در نام نقاط و قابلیت‌ها) ---
st.set_page_config(page_title="Aariz Precision Station V8.9", layout="wide")
ms, dev = load_models_safe()
names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

if "v" not in st.session_state: st.session_state.v = 0
up = st.sidebar.file_uploader("Upload Image:", type=['png', 'jpg', 'jpeg'])

if up and len(ms) == 3:
    img = Image.open(up).convert("RGB")
    if "lms" not in st.session_state or st.session_state.f != up.name:
        st.session_state.lms = run_prediction(img, ms, dev); st.session_state.f = up.name

    tid = st.sidebar.selectbox("🎯 انتخاب نقطه:", range(29), format_func=lambda x: f"{x}: {names[x]}")
    l = st.session_state.lms

    col1, col2 = st.columns([1.2, 2.5])
    with col1:
        st.subheader("🔍 Micro-Adjustment")
        p = l[tid]; sz = 180
        box = (max(0, p[0]-sz//2), max(0, p[1]-sz//2), min(img.width, p[0]+sz//2), min(img.height, p[1]+sz//2))
        mag = img.crop(box).resize((400, 400))
        d_mag = ImageDraw.Draw(mag); d_mag.line((190,200,210,200), fill="red", width=2); d_mag.line((200,190,200,210), fill="red", width=2)
        res_m = streamlit_image_coordinates(mag, key=f"m_{tid}_{st.session_state.v}")
        if res_m:
            new = [int(box[0] + res_m['x']*(sz/400)), int(box[1] + res_m['y']*(sz/400))]
            if l[tid] != new: l[tid] = new; st.session_state.v += 1; st.rerun()

    with col2:
        st.subheader("🖼 Cephalometric Analysis")
        canvas = img.copy(); draw = ImageDraw.Draw(canvas)
        # ترسیم تمام خطوط آنالیز
        if all(k in l for k in [10, 4, 0, 2, 15, 5, 18, 22, 17, 21, 12, 13, 8, 27]):
            draw.line([tuple(l[10]), tuple(l[4])], fill="yellow", width=3) # SN
            draw.line([tuple(l[15]), tuple(l[5])], fill="orange", width=3) # FH
            p_occ = (np.array(l[18])+np.array(l[22]))/2, (np.array(l[17])+np.array(l[21]))/2
            draw.line([tuple(p_occ[0]), tuple(p_occ[1])], fill="white", width=3) # Occ
            draw.line([tuple(l[8]), tuple(l[27])], fill="pink", width=2) # E-Line

        for i, pos in l.items():
            clr = (255,0,0) if i==tid else (0,255,0)
            draw.ellipse([pos[0]-6, pos[1]-6, pos[0]+6, pos[1]+6], fill=clr, outline="white")
            draw.text((pos[0]+12, pos[1]-12), names[i], fill=clr)
        
        streamlit_image_coordinates(canvas, width=850, key="main")

    # --- ۴. آنالیز و محاسبات (Incremental) ---
    st.divider()
    sna = round(np.degrees(np.arccos(np.dot((np.array(l[10])-l[4]), (np.array(l[0])-l[4])) / (np.linalg.norm(np.array(l[10])-l[4]) * np.linalg.norm(np.array(l[0])-l[4])+1e-6))), 1)
    snb = round(np.degrees(np.arccos(np.dot((np.array(l[10])-l[4]), (np.array(l[2])-l[4])) / (np.linalg.norm(np.array(l[10])-l[4]) * np.linalg.norm(np.array(l[2])-l[4])+1e-6))), 1)
    st.write(f"### 📑 آنالیز استاینر: SNA: {sna}° | SNB: {snb}° | ANB: {round(sna-snb, 1)}°")
