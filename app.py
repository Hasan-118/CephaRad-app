import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os, gdown, gc, datetime, io
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates
from fpdf import FPDF

# --- ۱. معماری مرجع Aariz (Gold Standard) ---
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

# --- ۲. لودر مدل با اصلاح خطا (Fix RuntimeError) ---
@st.cache_resource
def load_models_safe():
    model_ids = {
        'checkpoint_unet_clinical.pth': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks', 
        'specialist_pure_model.pth': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU', 
        'tmj_specialist_model.pth': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'
    }
    dev = torch.device("cpu"); ms = []
    for f, fid in model_ids.items():
        if not os.path.exists(f):
            gdown.download(f'https://drive.google.com/uc?id={fid}', f, quiet=True)
        try:
            m = CephaUNet(n_landmarks=29).to(dev)
            ckpt = torch.load(f, map_location=dev, weights_only=False)
            state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            # اصلاح کلیدها برای مدل‌های آموزش دیده با DataParallel
            clean_state = {k.replace('module.', ''): v for k, v in state.items()}
            m.load_state_dict(clean_state, strict=False)
            m.eval(); ms.append(m)
        except Exception as e: st.error(f"Error loading {f}: {e}")
    gc.collect(); return ms, dev

# --- ۳. پیش‌بینی بهینه (Fast Inference) ---
def predict_fast(img_pil, ms, dev):
    W, H = img_pil.size; ratio = 512 / max(W, H)
    img_rs = img_pil.convert('L').resize((int(W*ratio), int(H*ratio)), Image.NEAREST)
    canvas = Image.new("L", (512, 512)); px, py = (512-img_rs.width)//2, (512-img_rs.height)//2
    canvas.paste(img_rs, (px, py))
    tensor = transforms.ToTensor()(canvas).unsqueeze(0).to(dev)
    
    ANT_IDX, POST_IDX = {10, 14, 9, 5, 28, 20}, {7, 11, 12, 15}
    res = {}
    with torch.no_grad():
        outs = [m(tensor)[0].cpu().numpy() for m in ms]
        for i in range(29):
            # انتخاب مدل بر اساس تخصص
            m_idx = 1 if i in ANT_IDX else (2 if i in POST_IDX else 0)
            y, x = divmod(np.argmax(outs[m_idx][i]), 512)
            res[i] = [int((x - px) / ratio), int((y - py) / ratio)]
    return res

# --- ۴. رابط کاربری (UI) ---
st.set_page_config(page_title="Aariz V7.8 Turbo", layout="wide")
models, device = load_models_safe()
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

# سایدبار
st.sidebar.header("📋 مشخصات بیمار")
p_name = st.sidebar.text_input("نام بیمار:", "P-100")
doc_name = st.sidebar.text_input("پزشک:", "Dr. Aariz")
gender = st.sidebar.radio("جنسیت:", ["Male", "Female"])
pixel_size = st.sidebar.number_input("Pixel Size (mm):", 0.01, 1.0, 0.1, format="%.4f")
target_idx = st.sidebar.selectbox("🎯 انتخاب لندمارک:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")
uploaded_file = st.sidebar.file_uploader("آپلود تصویر سفالومتری", type=['png', 'jpg', 'jpeg'])

if uploaded_file and len(models) == 3:
    if "lms" not in st.session_state or st.session_state.file_id != uploaded_file.name:
        raw_img = Image.open(uploaded_file).convert("RGB")
        st.session_state.img = raw_img
        st.session_state.lms = predict_fast(raw_img, models, device)
        st.session_state.file_id = uploaded_file.name
        st.session_state.v = 0

    img = st.session_state.img; W, H = img.size
    col1, col2 = st.columns([1.2, 2.5])

    with col1:
        st.subheader("🔍 Micro-Adjustment")
        cur = st.session_state.lms[target_idx]
        # مگنیفایر سریع
        box = 120
        left, top = max(0, cur[0]-box), max(0, cur[1]-box)
        crop = img.crop((left, top, min(W, cur[0]+box), min(H, cur[1]+box))).resize((400, 400), Image.NEAREST)
        draw_m = ImageDraw.Draw(crop)
        draw_m.line((190, 200, 210, 200), fill="red", width=2); draw_m.line((200, 190, 200, 210), fill="red", width=2)
        res_m = streamlit_image_coordinates(crop, key=f"m_{target_idx}_{st.session_state.v}")
        if res_m:
            scale = (2*box)/400
            new_c = [int(left + (res_m['x'] * scale)), int(top + (res_m['y'] * scale))]
            if new_c != st.session_state.lms[target_idx]:
                st.session_state.lms[target_idx] = new_c
                st.session_state.v += 1; st.rerun()

    with col2:
        st.subheader("🖼 Cephalometric Trace")
        # رندر بهینه تصویر اصلی
        disp_scale = 850 / W
        disp_img = img.resize((850, int(H * disp_scale)), Image.NEAREST)
        draw = ImageDraw.Draw(disp_img)
        for i, p in st.session_state.lms.items():
            color, r = ((255,0,0), 8) if i == target_idx else ((0,255,0), 4)
            draw.ellipse([p[0]*disp_scale-r, p[1]*disp_scale-r, p[0]*disp_scale+r, p[1]*disp_scale+r], fill=color)
        
        res_main = streamlit_image_coordinates(disp_img, width=850, key=f"main_{st.session_state.v}")
        if res_main:
            new_c = [int(res_main['x'] / disp_scale), int(res_main['y'] / disp_scale)]
            if new_c != st.session_state.lms[target_idx]:
                st.session_state.lms[target_idx] = new_c
                st.session_state.v += 1; st.rerun()

    # --- ۵. بخش گزارش (On Demand) ---
    st.divider()
    if st.button("📄 تولید گزارش بالینی و PDF"):
        l = st.session_state.lms
        def get_ang(p1, p2, p3):
            v1, v2 = np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)
            norm = np.linalg.norm(v1)*np.linalg.norm(v2)
            return round(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/norm, -1, 1))), 2)
        
        sna = get_ang(l[10], l[4], l[0]); snb = get_ang(l[10], l[4], l[2])
        anb = round(sna - snb, 2)
        st.write(f"SNA: {sna} | SNB: {snb} | **ANB: {anb}**")
        # در اینجا می‌توانید کد تولید PDF قبلی را برای خروجی نهایی اضافه کنید.
