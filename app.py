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

# --- ۱. معماری مدل (بدون تغییر) ---
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

# --- ۲. بارگذاری و پیش‌بینی ---
@st.cache_resource
def load_aariz_models():
    model_ids = {'checkpoint_unet_clinical.pth': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks', 
                 'specialist_pure_model.pth': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU', 
                 'tmj_specialist_model.pth': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'}
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

def run_precise_prediction(img_pil, models, device):
    ow, oh = img_pil.size; img_gray = img_pil.convert('L'); ratio = 512 / max(ow, oh)
    nw, nh = int(ow * ratio), int(oh * ratio); img_rs = img_gray.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("L", (512, 512)); px, py = (512 - nw) // 2, (512 - nh) // 2
    canvas.paste(img_rs, (px, py)); input_tensor = transforms.ToTensor()(canvas).unsqueeze(0).to(device)
    with torch.no_grad(): outs = [m(input_tensor)[0].cpu().numpy() for m in models]
    ANT_IDX, POST_IDX = [10, 14, 9, 5, 28, 20], [7, 11, 12, 15]
    coords = {}
    for i in range(29):
        hm = outs[1][i] if i in ANT_IDX else (outs[2][i] if i in POST_IDX else outs[0][i])
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        coords[i] = [int((x - px) / ratio), int((y - py) / ratio)]
    gc.collect(); return coords

# --- ۳. رابط کاربری (UI) ---
st.set_page_config(page_title="Aariz Precision Station V7.8", layout="wide")
models, device = load_aariz_models()
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

if "v" not in st.session_state: st.session_state.v = 0

st.sidebar.header("📏 تنظیمات آنالیز")
gender = st.sidebar.radio("جنسیت:", ["Male", "Female"])
pixel_size = st.sidebar.number_input("Pixel Size (mm):", 0.01, 1.0, 0.1, format="%.4f")
text_scale = st.sidebar.slider("اندازه نام لندمارک:", 1, 10, 3)

uploaded_file = st.sidebar.file_uploader("آپلود تصویر:", type=['png', 'jpg', 'jpeg'])

if uploaded_file and len(models) == 3:
    raw = Image.open(uploaded_file).convert("RGB"); W, H = raw.size
    if "lms" not in st.session_state or st.session_state.get("fid") != uploaded_file.name:
        st.session_state.lms = run_precise_prediction(raw, models, device)
        st.session_state.fid = uploaded_file.name

    t_idx = st.sidebar.selectbox("🎯 لندمارک فعال:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")
    lms = st.session_state.lms

    col1, col2 = st.columns([1.2, 2.5])
    with col1:
        st.subheader("🔍 Magnifier")
        p = lms[t_idx]; b = 90 # Box size
        # اصلاح امنیتی کراپ برای جلوگیری از خطا
        left, top = max(0, min(p[0]-b, W-2*b)), max(0, min(p[1]-b, H-2*b))
        crop = raw.crop((left, top, left+2*b, top+2*b)).resize((400, 400), Image.LANCZOS)
        m_draw = ImageDraw.Draw(crop)
        m_draw.line((190, 200, 210, 200), fill="red", width=2); m_draw.line((200, 190, 200, 210), fill="red", width=2)
        
        res_m = streamlit_image_coordinates(crop, key=f"m_{t_idx}_{st.session_state.v}")
        if res_m:
            sc_m = (2*b) / 400
            new_p = [int(left + (res_m["x"] * sc_m)), int(top + (res_m["y"] * sc_m))]
            if lms[t_idx] != new_p:
                lms[t_idx] = new_p; st.session_state.v += 1; st.rerun()

    with col2:
        st.subheader("🖼 Cephalogram Analysis")
        disp = raw.copy(); draw = ImageDraw.Draw(disp)
        # رسم خطوط آنالیز
        if all(k in lms for k in [10, 4, 0, 2, 15, 5, 14, 3]):
            draw.line([tuple(lms[10]), tuple(lms[4])], fill="yellow", width=3) # S-N
            draw.line([tuple(lms[15]), tuple(lms[5])], fill="orange", width=3) # FH
        
        for i, pos in lms.items():
            clr = (255, 0, 0) if i == t_idx else (0, 255, 0)
            r = 10 if i == t_idx else 6
            draw.ellipse([pos[0]-r, pos[1]-r, pos[0]+r, pos[1]+r], fill=clr, outline="white", width=2)
            # رندر متن لندمارک
            name = landmark_names[i]
            t_img = Image.new('RGBA', (len(name)*10, 15), (0,0,0,0))
            ImageDraw.Draw(t_img).text((0,0), name, fill=clr)
            t_img = t_img.resize((int(t_img.width*text_scale/2), int(t_img.height*text_scale/2)))
            disp.paste(t_img, (pos[0]+12, pos[1]-10), t_img)

        res_main = streamlit_image_coordinates(disp, width=850, key=f"main_{st.session_state.v}")
        if res_main:
            sc_main = W / 850
            new_p = [int(res_main["x"] * sc_main), int(res_main["y"] * sc_main)]
            if lms[t_idx] != new_p:
                lms[t_idx] = new_p; st.session_state.v += 1; st.rerun()

    # --- ۴. بخش گزارش بالینی ---
    st.divider()
    def get_ang(p1, p2, p3):
        v1, v2 = np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)
        return round(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1, 1))), 2)

    sna = get_ang(lms[10], lms[4], lms[0]); snb = get_ang(lms[10], lms[4], lms[2])
    anb = round(sna - snb, 2)
    
    st.header(f"📑 Clinical Report ({gender})")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("ANB Angle", f"{anb}°", f"SNA: {sna} / SNB: {snb}")
        diag = "Class II" if anb > 4 else "Class III" if anb < 0 else "Class I"
        st.info(f"**Skeletal Diagnosis:** {diag}")
    with c2:
        st.success("Analysis Complete. Use the magnifier to refine points if needed.")
