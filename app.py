import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import gdown
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۱. معماری و لودر (بدون تغییر طبق Gold Standard) ---
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

@st.cache_resource
def load_aariz_models():
    model_ids = {'checkpoint_unet_clinical.pth': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks', 'specialist_pure_model.pth': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU', 'tmj_specialist_model.pth': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'}
    device = torch.device("cpu")
    loaded_models = []
    for f, fid in model_ids.items():
        if not os.path.exists(f): gdown.download(f'https://drive.google.com/uc?id={fid}', f, quiet=True)
        m = CephaUNet(n_landmarks=29).to(device)
        ckpt = torch.load(f, map_location=device)
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        m.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=False)
        m.eval(); loaded_models.append(m)
    return loaded_models, device

def run_precise_prediction(img_pil, models, device):
    ow, oh = img_pil.size
    img_gray = img_pil.convert('L')
    ratio = 512 / max(ow, oh)
    nw, nh = int(ow * ratio), int(oh * ratio)
    img_rs = img_gray.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("L", (512, 512))
    px, py = (512 - nw) // 2, (512 - nh) // 2
    canvas.paste(img_rs, (px, py))
    input_tensor = transforms.ToTensor()(canvas).unsqueeze(0).to(device)
    with torch.no_grad(): outs = [m(input_tensor)[0].cpu().numpy() for m in models]
    ANT_IDX, POST_IDX = [10, 14, 9, 5, 28, 20], [7, 11, 12, 15]
    coords = {}
    for i in range(29):
        hm = outs[1][i] if i in ANT_IDX else (outs[2][i] if i in POST_IDX else outs[0][i])
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        coords[i] = [int((x - px) / ratio), int((y - py) / ratio)]
    return coords

# --- ۲. محاسبات هندسی Downs ---
def get_angle(p1, p2, p3):
    v1, v2 = np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)
    norm = np.linalg.norm(v1)*np.linalg.norm(v2)
    if norm == 0: return 0
    return round(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/norm, -1, 1))), 2)

def calculate_downs_skeletal(l):
    try:
        # FH Plane: Po(15) to Or(5)
        # Facial Plane: N(4) to Pog(6)
        # Convexity: N(4)-A(0) and A(0)-Pog(6)
        
        facial_angle = get_angle(l[15], l[5], l[6]) # تقریب هندسی FH-NPog
        
        # برای Convexity، زاویه مکمل در نظر گرفته می‌شود (انحراف از خط مستقیم)
        v1 = np.array(l[4]) - np.array(l[0])
        v2 = np.array(l[6]) - np.array(l[0])
        convexity = get_angle(l[4], l[0], l[6])
        
        return facial_angle, convexity
    except: return 0, 0

# --- ۳. رابط کاربری (UI) ---
st.set_page_config(page_title="Aariz Precision Station V5.0", layout="wide")
models, device = load_aariz_models()
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

if "click_version" not in st.session_state: st.session_state.click_version = 0
st.sidebar.header("📏 تنظیمات کالیبراسیون")
pixel_size = st.sidebar.number_input("Pixel Size (mm/px):", 0.01, 1.0, 0.1, 0.001, format="%.4f")
text_scale = st.sidebar.slider("🔤 مقیاس ابعاد نام:", 1, 10, 3)

uploaded_file = st.sidebar.file_uploader("آپلود تصویر:", type=['png', 'jpg', 'jpeg'])

if uploaded_file and len(models) == 3:
    raw_img = Image.open(uploaded_file).convert("RGB")
    W, H = raw_img.size
    if "lms" not in st.session_state or st.session_state.get("file_id") != uploaded_file.name:
        st.session_state.initial_lms = run_precise_prediction(raw_img, models, device)
        st.session_state.lms = st.session_state.initial_lms.copy()
        st.session_state.file_id = uploaded_file.name

    target_idx = st.sidebar.selectbox("🎯 انتخاب لندمارک:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")
    
    col1, col2 = st.columns([1.2, 2.5])
    with col1:
        st.subheader("🔍 Micro-Adjustment")
        l_pos = st.session_state.lms[target_idx]
        size_m = 180 
        left, top = max(0, min(int(l_pos[0]-size_m//2), W-size_m)), max(0, min(int(l_pos[1]-size_m//2), H-size_m))
        mag_crop = raw_img.crop((left, top, left+size_m, top+size_m)).resize((400, 400), Image.LANCZOS)
        res_mag = streamlit_image_coordinates(mag_crop, key=f"mag_{target_idx}_{st.session_state.click_version}")
        if res_mag:
            scale_mag = size_m / 400
            new_c = [int(left + (res_mag["x"] * scale_mag)), int(top + (res_mag["y"] * scale_mag))]
            if st.session_state.lms[target_idx] != new_c:
                st.session_state.lms[target_idx] = new_c
                st.session_state.click_version += 1
                st.rerun()

    with col2:
        st.subheader("🖼 نمای گرافیکی (Steiner + Wits + Downs)")
        draw_img = raw_img.copy(); draw = ImageDraw.Draw(draw_img); l = st.session_state.lms
        
        # ۱. Steiner & Wits (از Gold Standard)
        if all(k in l for k in [10, 4, 0, 2, 22, 18, 21, 17]):
            draw.line([tuple(l[10]), tuple(l[4])], fill="yellow", width=3) # SN
            p_occ_p, p_occ_a = (np.array(l[18])+np.array(l[22]))/2, (np.array(l[17])+np.array(l[21]))/2
            draw.line([tuple(p_occ_p), tuple(p_occ_a)], fill="white", width=3) # Functional Occlusal
            
        # ۲. Downs Lines (جدید)
        if all(k in l for k in [15, 5, 4, 6, 0]):
            draw.line([tuple(l[15]), tuple(l[5])], fill="orange", width=3) # Frankfort Plane (Po-Or)
            draw.line([tuple(l[4]), tuple(l[6])], fill="blue", width=2) # N-Pog
            draw.line([tuple(l[4]), tuple(l[0]), tuple(l[6])], fill="green", width=2) # Convexity (N-A-Pog)

        for i, pos in l.items():
            color = (255, 0, 0) if i == target_idx else (0, 255, 0)
            draw.ellipse([pos[0]-6, pos[1]-6, pos[0]+6, pos[1]+6], fill=color, outline="white")
            
        res_main = streamlit_image_coordinates(draw_img, width=850, key=f"main_{st.session_state.click_version}")
        if res_main:
            c_scale = W / 850
            new_c = [int(res_main["x"] * c_scale), int(res_main["y"] * c_scale)]
            if st.session_state.lms[target_idx] != new_c:
                st.session_state.lms[target_idx] = new_c
                st.session_state.click_version += 1
                st.rerun()

    # --- نتایج آنالیز ترکیبی ---
    st.divider()
    sna, snb = get_angle(l[10], l[4], l[0]), get_angle(l[10], l[4], l[2])
    f_angle, convexity = calculate_downs_skeletal(l)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SNA / SNB", f"{sna}° / {snb}°", f"ANB: {round(sna-snb,2)}°")
    c2.metric("Wits (Func.)", f"{round(((np.dot(np.array(l[0])-(np.array(l[18])+np.array(l[22]))/2, (np.array(l[17])+np.array(l[21]))/2-(np.array(l[18])+np.array(l[22]))/2)/(np.linalg.norm((np.array(l[17])+np.array(l[21]))/2-(np.array(l[18])+np.array(l[22]))/2)+1e-6)) - (np.dot(np.array(l[2])-(np.array(l[18])+np.array(l[22]))/2, (np.array(l[17])+np.array(l[21]))/2-(np.array(l[18])+np.array(l[22]))/2)/(np.linalg.norm((np.array(l[17])+np.array(l[21]))/2-(np.array(l[18])+np.array(l[22]))/2)+1e-6)))*pixel_size, 2)} mm")
    c3.metric("Facial Angle (Downs)", f"{f_angle}°", "Normal: 87.8°")
    c4.metric("Convexity (Downs)", f"{convexity}°", "Normal: 0°")
