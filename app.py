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

# --- ۱. معماری مرجع (بدون تغییر) ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
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

# --- ۲. لودر بهینه (Caching لایه‌ای) ---
@st.cache_resource
def load_models_fast():
    ids = {'checkpoint_unet_clinical.pth': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks', 'specialist_pure_model.pth': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU', 'tmj_specialist_model.pth': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'}
    dev = torch.device("cpu"); ms = []
    for f, fid in ids.items():
        if not os.path.exists(f): gdown.download(f'https://drive.google.com/uc?id={fid}', f, quiet=True)
        m = CephaUNet().to(dev); ck = torch.load(f, map_location=dev)
        m.load_state_dict({k.replace('module.', ''): v for k, v in (ck['model_state_dict'] if 'model_state_dict' in ck else ck).items()}, strict=False)
        m.eval(); ms.append(m)
    return ms, dev

def predict_fast(img, ms, dev):
    ow, oh = img.size; r = 512 / max(ow, oh); nw, nh = int(ow*r), int(oh*r)
    canvas = Image.new("L", (512, 512)); canvas.paste(img.convert('L').resize((nw, nh), Image.LANCZOS), ((512-nw)//2, (512-nh)//2))
    t = transforms.ToTensor()(canvas).unsqueeze(0).to(dev)
    with torch.no_grad(): outs = [m(t)[0].cpu().numpy() for m in ms]
    lms = {}
    for i in range(29):
        h = outs[1][i] if i in [10, 14, 9, 5, 28, 20] else (outs[2][i] if i in [7, 11, 12, 15] else outs[0][i])
        y, x = np.unravel_index(np.argmax(h), (512, 512))
        lms[i] = [int((x - (512-nw)//2)/r), int((y - (512-nh)//2)/r)]
    return lms

# --- ۳. رابط کاربری (UI) با مدیریت Rerun ---
st.set_page_config(layout="wide")
ms, dev = load_models_fast()
names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

if "v" not in st.session_state: st.session_state.v = 0

up = st.sidebar.file_uploader("Cephalogram:", type=['png','jpg'])
px_size = st.sidebar.number_input("Pixel Size:", 0.01, 1.0, 0.1)

if up:
    img = Image.open(up).convert("RGB")
    if "lms" not in st.session_state or st.session_state.f != up.name:
        st.session_state.lms = predict_fast(img, ms, dev); st.session_state.f = up.name

    tid = st.sidebar.selectbox("Active Point:", range(29), format_func=lambda x: f"{x}: {names[x]}")
    
    col1, col2 = st.columns([1, 2])
    l = st.session_state.lms
    
    with col1: # میکرو-تنظیم بسیار سریع
        p = l[tid]; sz = 160
        box = (max(0, p[0]-sz//2), max(0, p[1]-sz//2), min(img.width, p[0]+sz//2), min(img.height, p[1]+sz//2))
        mag = img.crop(box).resize((300, 300))
        draw_m = ImageDraw.Draw(mag); draw_m.line((140, 150, 160, 150), fill="red"); draw_m.line((150, 140, 150, 160), fill="red")
        res = streamlit_image_coordinates(mag, key=f"m_{tid}_{st.session_state.v}")
        if res:
            new = [int(box[0] + res['x']*(sz/300)), int(box[1] + res['y']*(sz/300))]
            if l[tid] != new: l[tid] = new; st.session_state.v += 1; st.rerun()

    with col2: # نمایش خطوط با حداقل محاسبات در هر فریم
        canvas = img.copy(); draw = ImageDraw.Draw(canvas)
        # فقط خطوط اصلی برای جلوگیری از افت سرعت
        if all(k in l for k in [10,4,0,2,15,5,18,22,17,21]):
            draw.line([tuple(l[10]), tuple(l[4])], fill="yellow", width=2) # SN
            draw.line([tuple(l[15]), tuple(l[5])], fill="orange", width=2) # FH
            p_occ = (np.array(l[18])+np.array(l[22]))/2, (np.array(l[17])+np.array(l[21]))/2
            draw.line([tuple(p_occ[0]), tuple(p_occ[1])], fill="white", width=2) # Occ

        for i, pos in l.items():
            draw.ellipse([pos[0]-5, pos[1]-5, pos[0]+5, pos[1]+5], fill=(0,255,0) if i!=tid else (255,0,0))
        
        streamlit_image_coordinates(canvas, width=800, key="main")

    # --- ۴. آنالیز (Wits & McNamara) ---
    st.divider()
    def get_ang(p1, p2, p3):
        v1, v2 = np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)
        return round(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1, 1))), 1)

    sna, snb = get_ang(l[10],l[4],l[0]), get_ang(l[10],l[4],l[2])
    anb = round(sna-snb, 1)
    # Wits
    p_occ = (np.array(l[18])+np.array(l[22]))/2, (np.array(l[17])+np.array(l[21]))/2
    v_occ = (p_occ[1]-p_occ[0])/np.linalg.norm(p_occ[1]-p_occ[0])
    wits = round((np.dot(np.array(l[0])-p_occ[0], v_occ) - np.dot(np.array(l[2])-p_occ[0], v_occ))*px_size, 1)
    
    st.subheader(f"📊 آنالیز سریع: ANB: {anb} | Wits: {wits}mm")
    
    if st.button("📄 صدور طرح درمان مفصل"):
        diag = "Class II" if wits > 2 else "Class III" if wits < -2 else "Class I"
        st.write(f"**طرح درمان:** با توجه به ویتز {wits} و رابطه {diag}، اولویت با جابجایی اسکلتال است.")
