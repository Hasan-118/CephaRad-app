import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import gdown
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۱. ساختار مدل مرجع (قفل شده برای حفظ وزن‌ها) ---
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

# --- ۲. لودر سیستم و توابع ترسیم ---
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

def get_safe_magnifier(img, coord, size=120):
    w, h = img.size
    x, y = coord
    left = max(0, min(int(x - size//2), w - size))
    top = max(0, min(int(y - size//2), h - size))
    crop = img.crop((left, top, left + size, top + size)).resize((400, 400), Image.LANCZOS)
    draw = ImageDraw.Draw(crop)
    draw.line((180, 200, 220, 200), fill="#FF0000", width=3) # نشانگر مرکز افقی
    draw.line((200, 180, 200, 220), fill="#FF0000", width=3) # نشانگر مرکز عمودی
    return crop, (left, top)

# --- ۳. رابط کاربری اصلی ---
st.set_page_config(page_title="Aariz Precision V3.9", layout="wide")
models, device = load_aariz_system()
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

if "click_version" not in st.session_state: st.session_state.click_version = 0

uploaded_file = st.sidebar.file_uploader("آپلود تصویر سفالومتری:", type=['png', 'jpg', 'jpeg'])

if uploaded_file and len(models) == 3:
    raw_img = Image.open(uploaded_file).convert("RGB")
    W, H = raw_img.size
    
    if "lms" not in st.session_state or st.session_state.get("file_id") != uploaded_file.name:
        with st.spinner("AI Ensemble Analysis..."):
            img_input = raw_img.convert('L').resize((512, 512), Image.LANCZOS)
            t = transforms.ToTensor()(img_input).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = [m(t)[0].cpu().numpy() for m in models]
            coords = {}
            sx, sy = W / 512.0, H / 512.0
            ANT_IDX, TMJ_IDX = [1, 20, 21, 22, 24, 25, 26, 28], [11, 12, 15, 16]
            for i in range(29):
                hm = preds[1][i] if i in ANT_IDX else (preds[2][i] if i in TMJ_IDX else preds[0][i])
                y, x = np.unravel_index(np.argmax(hm), hm.shape)
                coords[i] = [int(x * sx), int(y * sy)]
            st.session_state.lms = coords
            st.session_state.file_id = uploaded_file.name

    target_idx = st.sidebar.selectbox("🎯 انتخاب لندمارک فعال:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")

    col1, col2 = st.columns([1.2, 2])
    with col1:
        st.subheader("🔍 Micro-Adjustment")
        mag_img, (off_x, off_y) = get_safe_magnifier(raw_img, st.session_state.lms[target_idx])
        res_mag = streamlit_image_coordinates(mag_img, key=f"mag_{target_idx}_{st.session_state.click_version}")
        if res_mag:
            scale = 120 / 400
            new_x, new_y = int(off_x + (res_mag["x"] * scale)), int(off_y + (res_mag["y"] * scale))
            if st.session_state.lms[target_idx] != [new_x, new_y]:
                st.session_state.lms[target_idx] = [new_x, new_y]
                st.session_state.click_version += 1
                st.rerun()

    with col2:
        st.subheader("🖼 Full View (Steiner Analysis)")
        draw_img = raw_img.copy()
        draw = ImageDraw.Draw(draw_img)
        l = st.session_state.lms
        
        # ترسیم خطوط Steiner (S-N, N-A, N-B)
        if all(k in l for k in [10, 4, 0, 2]): # S, N, A, B
            draw.line([tuple(l[10]), tuple(l[4])], fill="#FFFF00", width=3) # S-N Line
            draw.line([tuple(l[4]), tuple(l[0])], fill="#00FFFF", width=3) # N-A Line
            draw.line([tuple(l[4]), tuple(l[2])], fill="#FF00FF", width=3) # N-B Line

        for i, pos in l.items():
            color = "red" if i == target_idx else "#00FF00"
            r = 15 if i == target_idx else 8
            draw.ellipse([pos[0]-r, pos[1]-r, pos[0]+r, pos[1]+r], fill=color, outline="white", width=2)
        
        res_main = streamlit_image_coordinates(draw_img, width=850, key="main_canvas")
        if res_main:
            c_scale = W / 850
            m_coord = [int(res_main["x"] * c_scale), int(res_main["y"] * c_scale)]
            if st.session_state.lms[target_idx] != m_coord:
                st.session_state.lms[target_idx] = m_coord
                st.session_state.click_version += 1
                st.rerun()

    # --- محاسبات نهایی ---
    st.divider()
    def get_ang(p1, p2, p3):
        v1, v2 = np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)
        n = np.linalg.norm(v1)*np.linalg.norm(v2)
        return round(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n if n>0 else 1), -1, 1))), 1)
    
    sna, snb = get_ang(l[10], l[4], l[0]), get_ang(l[10], l[4], l[2])
    c1, c2, c3 = st.columns(3)
    c1.metric("SNA (Maxilla Relationship)", f"{sna}°")
    c2.metric("SNB (Mandible Relationship)", f"{snb}°")
    c3.metric("ANB (Skeletal Class)", f"{round(sna-snb, 1)}°")
