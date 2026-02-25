import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import gdown
import gc
import io
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۱. معماری مرجع Aariz (ثابت) ---
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

# --- ۲. بارگذاری مدل‌ها و پیش‌بینی ---
@st.cache_resource
def load_aariz_models():
    model_ids = {'checkpoint_unet_clinical.pth': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks', 
                 'specialist_pure_model.pth': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU', 
                 'tmj_specialist_model.pth': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'}
    device = torch.device("cpu"); loaded_models = []
    for f, fid in model_ids.items():
        if not os.path.exists(f): gdown.download(f'https://drive.google.com/uc?id={fid}', f, quiet=True)
        try:
            m = CephaUNet(n_landmarks=29).to(device)
            ckpt = torch.load(f, map_location=device)
            state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            m.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=False)
            m.eval(); loaded_models.append(m)
        except: pass
    return loaded_models, device

def run_prediction(img_pil, models, device):
    ow, oh = img_pil.size; img_gray = img_pil.convert('L'); ratio = 512 / max(ow, oh)
    nw, nh = int(ow * ratio), int(oh * ratio); canvas = Image.new("L", (512, 512))
    px, py = (512-nw)//2, (512-nh)//2; canvas.paste(img_gray.resize((nw, nh), Image.LANCZOS), (px, py))
    input_tensor = transforms.ToTensor()(canvas).unsqueeze(0).to(device)
    with torch.no_grad(): outs = [m(input_tensor)[0].cpu().numpy() for m in models]
    ANT, POST = [10, 14, 9, 5, 28, 20], [7, 11, 12, 15]
    return {i: [int((np.unravel_index(np.argmax(outs[1][i] if i in ANT else (outs[2][i] if i in POST else outs[0][i])), (512,512))[1]-px)/ratio),
                int((np.unravel_index(np.argmax(outs[1][i] if i in ANT else (outs[2][i] if i in POST else outs[0][i])), (512,512))[0]-py)/ratio)] for i in range(29)}

# --- ۳. رابط کاربری اصلی ---
st.set_page_config(page_title="Aariz Precision Station V7.8.20", layout="wide")
models, device = load_aariz_models()
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

with st.sidebar:
    st.header("📏 تنظیمات")
    gender = st.radio("جنسیت:", ["آقا", "خانم"])
    px_size = st.number_input("Pixel Size (mm/px):", 0.01, 1.0, 0.1, format="%.4f")
    uploaded_file = st.file_uploader("آپلود تصویر:", type=['png', 'jpg', 'jpeg'])

if uploaded_file and len(models) == 3:
    raw_img = Image.open(uploaded_file).convert("RGB"); W, H = raw_img.size
    if "lms" not in st.session_state or st.session_state.get("f_id") != uploaded_file.name:
        st.session_state.lms = run_prediction(raw_img, models, device)
        st.session_state.f_id = uploaded_file.name
        st.session_state.ver = 0

    target = st.sidebar.selectbox("🎯 انتخاب نقطه:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("🔍 Micro-Adjustment")
        l_pos = st.session_state.lms[target]; crop_sz = 160
        left, top = max(0, min(int(l_pos[0]-crop_sz//2), W-crop_sz)), max(0, min(int(l_pos[1]-crop_sz//2), H-crop_sz))
        mag = raw_img.crop((left, top, left+crop_sz, top+crop_sz)).resize((400, 400))
        res_mag = streamlit_image_coordinates(mag, key=f"mag_{st.session_state.ver}")
        if res_mag:
            new_c = [int(left + res_mag["x"]*crop_sz/400), int(top + res_mag["y"]*crop_sz/400)]
            if st.session_state.lms[target] != new_c:
                st.session_state.lms[target] = new_c; st.session_state.ver += 1; st.rerun()

    with col2:
        st.subheader("🖼 آنالیز گرافیکی")
        draw_img = raw_img.copy(); draw = ImageDraw.Draw(draw_img); l = st.session_state.lms
        # رسم خطوط آنالیز
        lines = [(10,4,"yellow"), (4,0,"cyan"), (4,2,"magenta"), (15,5,"orange"), (14,3,"purple")]
        for p1, p2, c in lines: draw.line([tuple(l[p1]), tuple(l[p2])], fill=c, width=3)
        for i, pos in l.items():
            clr = (255,0,0) if i == target else (0,255,0)
            draw.ellipse([pos[0]-5, pos[1]-5, pos[0]+5, pos[1]+5], fill=clr)
        st.image(draw_img, use_column_width=True)

    # --- ۴. بخش گزارش تخصصی ---
    st.divider()
    def get_ang(p1, p2, p3):
        v1, v2 = np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)
        n = np.linalg.norm(v1)*np.linalg.norm(v2)
        return round(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n if n>0 else 1e-6), -1, 1))), 2)

    sna, snb = get_ang(l[10], l[4], l[0]), get_ang(l[10], l[4], l[2])
    anb = round(sna - snb, 2)
    diff_mcn = round((np.linalg.norm(np.array(l[12])-np.array(l[13])) - np.linalg.norm(np.array(l[12])-np.array(l[0]))) * px_size, 2)
    diag = "Class II" if anb > 4 else "Class III" if anb < 0 else "Class I"

    st.header(f"📑 گزارش تحلیلی Aariz ({diag})")
    m1, m2, m3 = st.columns(3)
    m1.metric("SNA (Maxilla)", f"{sna}°")
    m2.metric("SNB (Mandible)", f"{snb}°")
    m3.metric("ANB (Base)", f"{anb}°", diag, delta_color="inverse" if diag != "Class I" else "normal")
    
    st.write(f"**McNamara Diff:** {diff_mcn} mm (Target: 25-30mm)")
    
    if st.button("📥 ذخیره گزارش"):
        rep = f"Aariz Report\nGender: {gender}\nDiagnosis: {diag}\nSNA: {sna}\nSNB: {snb}\nANB: {anb}\nMcNamara: {diff_mcn}mm"
        st.download_button("Download TXT", rep, "Report.txt")

gc.collect()
