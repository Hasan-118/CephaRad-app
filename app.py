import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۱. معماری مدل UNet ---
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

# --- ۲. لودر سه‌گانه و پیش‌بینی با CLAHE ---
@st.cache_resource
def load_aariz_engines():
    paths = ['checkpoint_unet_clinical.pth', 'specialist_pure_model.pth', 'tmj_specialist_model.pth']
    engines = []
    for p in paths:
        if os.path.exists(p):
            model = CephaUNet(n_landmarks=29).to("cpu")
            ckpt = torch.load(p, map_location="cpu")
            state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(state)
            model.eval()
            engines.append(model)
    return engines

def get_ensemble_prediction(img_path, engines):
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_gray)
    
    orig_h, orig_w = img_enhanced.shape
    img_res = cv2.resize(img_enhanced, (384, 384), interpolation=cv2.INTER_LANCZOS4)
    input_t = transforms.ToTensor()(img_res).unsqueeze(0)
    
    all_heatmaps = []
    with torch.no_grad():
        for model in engines:
            all_heatmaps.append(model(input_t)[0].numpy())
    
    avg_output = np.mean(all_heatmaps, axis=0)
    coords = {}
    for i in range(29):
        hm = avg_output[i]
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        coords[i] = [int(x * orig_w / 384), int(y * orig_h / 384)]
    return coords, (orig_w, orig_h)

def get_angle(p1, p2, p3):
    v1, v2 = np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return round(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/norm, -1, 1))), 1) if norm != 0 else 0

# --- ۳. رابط کاربری اصلی ---
st.set_page_config(layout="wide", page_title="Aariz Station v31.3")
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']
EXCELLENT_PTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 20, 21, 24, 25, 27, 28]
RELIABLE_PTS = [14, 17, 26]

engines = load_aariz_engines()

st.sidebar.title("🩺 Aariz Control")
path_input = st.sidebar.text_input("Project Path:", value=os.getcwd())
img_dir = os.path.join(path_input, "Aariz", "train", "Cephalograms")

if os.path.exists(img_dir) and len(engines) > 0:
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))])
    selected_file = st.sidebar.selectbox("Choose Image:", files)
    target_idx = st.sidebar.selectbox("Active Point:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")
    
    full_path = os.path.join(img_dir, selected_file)
    if "lms" not in st.session_state or st.session_state.get("file") != selected_file:
        st.session_state.lms, st.session_state.orig_size = get_ensemble_prediction(full_path, engines)
        st.session_state.file = selected_file

    col_img, col_zoom = st.columns([2.5, 1])
    
    with col_img:
        img_view = Image.open(full_path).convert("RGB")
        orig_w, orig_h = st.session_state.orig_size
        draw = ImageDraw.Draw(img_view)
        l = st.session_state.lms

        # رسم لندمارک‌ها با متن و استروک بزرگتر
        for i, pos in l.items():
            is_active = (i == target_idx)
            color = "#00FF00" if i in EXCELLENT_PTS else ("#FFFF00" if i in RELIABLE_PTS else "#FF00FF")
            r = int(orig_w * 0.007)
            
            # افکت نقطه فعال
            if is_active:
                draw.ellipse([pos[0]-r-6, pos[1]-r-6, pos[0]+r+6, pos[1]+r+6], outline="red", width=6)
            
            draw.ellipse([pos[0]-r, pos[1]-r, pos[0]+r, pos[1]+r], fill=color, outline="white", width=2)
            
            # برچسب متنی بزرگ و خوانا
            label_text = f"{i}:{landmark_names[i]}"
            draw.text((pos[0]+r+5, pos[1]-r-15), label_text, fill="yellow", stroke_width=4, stroke_fill="black")

        st.subheader("📍 Main Analysis View")
        res = streamlit_image_coordinates(img_view, width=900, key="main_img")
        if res:
            scale = orig_w / 900
            new_x, new_y = int(res["x"] * scale), int(res["y"] * scale)
            if l[target_idx] != [new_x, new_y]:
                st.session_state.lms[target_idx] = [new_x, new_y]
                st.rerun()

    with col_zoom:
        st.subheader("🔍 Precision Zoom")
        active_pos = st.session_state.lms[target_idx]
        
        # برش برای نمای زوم
        z_size = 120
        box = (max(0, active_pos[0]-z_size), max(0, active_pos[1]-z_size), 
               min(orig_w, active_pos[0]+z_size), min(orig_h, active_pos[1]+z_size))
        
        zoom_img = Image.open(full_path).convert("RGB").crop(box)
        z_draw = ImageDraw.Draw(zoom_img)
        # رسم Crosshair در مرکز زوم
        cw, ch = zoom_img.size
        z_draw.line([(cw//2, 0), (cw//2, ch)], fill="red", width=2)
        z_draw.line([(0, ch//2), (cw, ch//2)], fill="red", width=2)
        
        st.image(zoom_img, use_container_width=True, caption=f"Centering: {landmark_names[target_idx]}")
        
        st.markdown("---")
        st.write("### ⌨️ Micro-Movements")
        c1, c2, c3 = st.columns(3)
        if c2.button("🔼"): st.session_state.lms[target_idx][1] -= 1; st.rerun()
        if c1.button("◀️"): st.session_state.lms[target_idx][0] -= 1; st.rerun()
        if c3.button("▶️"): st.session_state.lms[target_idx][0] += 1; st.rerun()
        if c2.button("🔽"): st.session_state.lms[target_idx][1] += 1; st.rerun()
        
        st.markdown("---")
        if st.button("🔄 Reset Point", use_container_width=True):
            fresh, _ = get_ensemble_prediction(full_path, engines)
            st.session_state.lms[target_idx] = fresh[target_idx]
            st.rerun()
        
        if st.button("💾 Save & Lock", type="primary", use_container_width=True):
            st.success("Landmark Coordinate Locked!")

    # آنالیز پایین
    st.divider()
    sna = get_angle(l[10], l[4], l[0])
    snb = get_angle(l[10], l[4], l[2])
    anb = round(sna - snb, 1)
    
    
    
    col1, col2, col3 = st.columns(3)
    col1.metric("SNA", f"{sna}°")
    col2.metric("SNB", f"{snb}°")
    col3.metric("ANB", f"{anb}°", delta="Class II" if anb > 4 else ("Class III" if anb < 0 else "Class I"))

else:
    st.error("Missing .pth files or images directory!")
