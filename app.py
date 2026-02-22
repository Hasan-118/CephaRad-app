import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import gdown
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۱. معماری مرجع Aariz ---
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

# --- ۲. لودر و توابع کمکی ---
@st.cache_resource
def load_aariz_models():
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
    with torch.no_grad():
        outs = [m(input_tensor)[0].cpu().numpy() for m in models]
    ANT_IDX, POST_IDX = [10, 14, 9, 5, 28, 20], [7, 11, 12, 15]
    coords = {}
    for i in range(29):
        hm = outs[1][i] if i in ANT_IDX else (outs[2][i] if i in POST_IDX else outs[0][i])
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        coords[i] = [int((x - px) / ratio), int((y - py) / ratio)]
    return coords

# --- ۳. رابط کاربری نهایی ---
st.set_page_config(page_title="Aariz Precision Station V4.3", layout="wide")
models, device = load_aariz_models()
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

# مقداردهی اولیه به متغیرهای کنترلی
if "click_version" not in st.session_state: st.session_state.click_version = 0
if "last_target" not in st.session_state: st.session_state.last_target = 0

# اسلایدر تنظیم سایز لندمارک و متن
label_size = st.sidebar.slider("📏 سایز نام و لندمارک:", 10, 80, 30)

uploaded_file = st.sidebar.file_uploader("آپلود تصویر:", type=['png', 'jpg', 'jpeg'])

if uploaded_file and len(models) == 3:
    raw_img = Image.open(uploaded_file).convert("RGB")
    W, H = raw_img.size
    
    if "lms" not in st.session_state or st.session_state.get("file_id") != uploaded_file.name:
        with st.spinner("استخراج هوشمند لندمارک‌ها..."):
            st.session_state.lms = run_precise_prediction(raw_img, models, device)
            st.session_state.file_id = uploaded_file.name

    target_idx = st.sidebar.selectbox("🎯 انتخاب لندمارک فعال:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")
    
    # جلوگیری از پرش: اگر لندمارک عوض شد، ورژن کلیک را بالا ببر تا مختصات قبلی پاک شود
    if st.session_state.last_target != target_idx:
        st.session_state.click_version += 1
        st.session_state.last_target = target_idx
        st.rerun()

    col1, col2 = st.columns([1.2, 2])
    
    # توابع ترسیم ذره‌بین و تصویر اصلی
    with col1:
        st.subheader("🔍 Micro-Adjustment")
        l_pos = st.session_state.lms[target_idx]
        # ایجاد ذره‌بین
        size = 120
        left, top = max(0, min(int(l_pos[0] - size//2), W - size)), max(0, min(int(l_pos[1] - size//2), H - size))
        mag_crop = raw_img.crop((left, top, left + size, top + size)).resize((400, 400), Image.LANCZOS)
        mag_draw = ImageDraw.Draw(mag_crop)
        mag_draw.line((180, 200, 220, 200), fill="red", width=3); mag_draw.line((200, 180, 200, 220), fill="red", width=3)
        
        res_mag = streamlit_image_coordinates(mag_crop, key=f"mag_{target_idx}_{st.session_state.click_version}")
        if res_mag:
            scale_mag = size / 400
            new_coord = [int(left + (res_mag["x"] * scale_mag)), int(top + (res_mag["y"] * scale_mag))]
            if st.session_state.lms[target_idx] != new_coord:
                st.session_state.lms[target_idx] = new_coord
                st.session_state.click_version += 1
                st.rerun()

    with col2:
        st.subheader("🖼 Full View & Steiner")
        draw_img = raw_img.copy()
        draw = ImageDraw.Draw(draw_img)
        l = st.session_state.lms
        
        # Steiner Lines
        if all(k in l for k in [10, 4, 0, 2]):
            draw.line([tuple(l[10]), tuple(l[4])], fill="yellow", width=4)
            draw.line([tuple(l[4]), tuple(l[0])], fill="cyan", width=4)
            draw.line([tuple(l[4]), tuple(l[2])], fill="magenta", width=4)

        # فونت پویا بر اساس اسلایدر
        try: font = ImageFont.truetype("arial.ttf", label_size)
        except: font = ImageFont.load_default()

        for i, pos in l.items():
            is_active = (i == target_idx)
            color = "red" if is_active else "#00FF00"
            # شعاع لندمارک بر اساس اسلایدر
            r = int(label_size / 2) if is_active else int(label_size / 4)
            draw.ellipse([pos[0]-r, pos[1]-r, pos[0]+r, pos[1]+r], fill=color, outline="white", width=2)
            
            # رسم نام با سایه
            draw.text((pos[0] + r + 5, pos[1] - r + 2), landmark_names[i], fill="black", font=font)
            draw.text((pos[0] + r + 3, pos[1] - r), landmark_names[i], fill=color, font=font)
        
        res_main = streamlit_image_coordinates(draw_img, width=850, key=f"main_{st.session_state.click_version}")
        if res_main:
            c_scale = W / 850
            m_coord = [int(res_main["x"] * c_scale), int(res_main["y"] * c_scale)]
            if st.session_state.lms[target_idx] != m_coord:
                st.session_state.lms[target_idx] = m_coord
                st.session_state.click_version += 1
                st.rerun()

    # --- Metrics ---
    st.divider()
    def get_ang(p1, p2, p3):
        v1, v2 = np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)
        n = np.linalg.norm(v1)*np.linalg.norm(v2)
        return round(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n if n>0 else 1), -1, 1))), 2)
    
    sna, snb = get_ang(l[10], l[4], l[0]), get_ang(l[10], l[4], l[2])
    c1, c2, c3 = st.columns(3)
    c1.metric("SNA", f"{sna}°")
    c2.metric("SNB", f"{snb}°")
    c3.metric("ANB", f"{round(sna-snb, 2)}°")
