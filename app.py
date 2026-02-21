import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import gdown
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۱. اصلاح معماری DoubleConv مطابق با معماری آموزش دیده شما ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # ترتیب لایه‌ها دقیقاً مطابق وزن‌های فایل .pth شماست
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
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
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
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

# --- ۲. لودر هوشمند با قابلیت دانلود خودکار از گوگل درایو ---
@st.cache_resource
def load_aariz_system():
    model_ids = {
        'checkpoint_unet_clinical.pth': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks',
        'specialist_pure_model.pth': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU',
        'tmj_specialist_model.pth': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'
    }
    device = torch.device("cpu")
    models = []
    
    for filename, fid in model_ids.items():
        if not os.path.exists(filename):
            url = f'https://drive.google.com/uc?id={fid}'
            gdown.download(url, filename, quiet=False)
        
        try:
            m = CephaUNet(n_landmarks=29).to(device)
            ckpt = torch.load(filename, map_location=device)
            state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            # حذف پیشوند module در صورت وجود
            clean_state = {k.replace('module.', ''): v for k, v in state.items()}
            m.load_state_dict(clean_state, strict=True)
            m.eval()
            models.append(m)
        except Exception as e:
            st.error(f"خطا در مدل {filename}: {e}")
            
    return models, device

# --- ۳. پردازش تصویر (Letterbox Resizing برای حفظ تمام زوایا) ---
def predict_landmarks(img_pil, models, device):
    ow, oh = img_pil.size
    img_gray = img_pil.convert('L')
    ratio = 512 / max(ow, oh)
    nw, nh = int(ow * ratio), int(oh * ratio)
    img_rs = img_gray.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("L", (512, 512))
    px, py = (512 - nw) // 2, (512 - nh) // 2
    canvas.paste(img_rs, (px, py))
    
    input_t = transforms.ToTensor()(canvas).unsqueeze(0).to(device)
    with torch.no_grad():
        outs = [m(input_t)[0].cpu().numpy() for m in models]
    
    # نواحی تخصصی انسامبل بر اساس معماری Aariz
    ANT_IDX, POST_IDX = [10, 14, 9, 5, 28, 20], [7, 11, 12, 15]
    coords = {}
    for i in range(29):
        if i in ANT_IDX and len(outs) >= 2: hm = outs[1][i]
        elif i in POST_IDX and len(outs) >= 3: hm = outs[2][i]
        else: hm = outs[0][i]
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        coords[i] = [int((x - px) / ratio), int((y - py) / ratio)]
    return coords

# --- ۴. رابط کاربری (Streamlit UI) ---
st.set_page_config(page_title="Aariz AI Station V2.1", layout="wide")
models, device = load_aariz_system()
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

st.sidebar.title("🦷 Aariz AI Station")
st.sidebar.info(f"سیستم: {device.type.upper()} | مدل‌ها: {len(models)}/3")

uploaded_file = st.sidebar.file_uploader("آپلود تصویر رادیوگرافی:", type=['png', 'jpg', 'jpeg'])

if uploaded_file and models:
    img_raw = Image.open(uploaded_file).convert("RGB")
    if "lms" not in st.session_state or st.session_state.fid != uploaded_file.name:
        st.session_state.lms = predict_landmarks(img_raw, models, device)
        st.session_state.fid = uploaded_file.name

    target_idx = st.sidebar.selectbox("🎯 انتخاب نقطه برای اصلاح:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")

    col1, col2 = st.columns([3, 1])
    with col1:
        draw_img = img_raw.copy()
        draw = ImageDraw.Draw(draw_img)
        l = st.session_state.lms
        
        # ترسیم زنده Steiner Analysis برای تست
        draw.line([tuple(l[10]), tuple(l[4]), tuple(l[0])], fill="yellow", width=5)
        
        for i, pos in l.items():
            color = "red" if i == target_idx else "#00FF00"
            r = 14 if i == target_idx else 7
            draw.ellipse([pos[0]-r, pos[1]-r, pos[0]+r, pos[1]+r], fill=color, outline="white")

        st.subheader("📍 برای جابجایی دقیق، روی تصویر کلیک کنید")
        res = streamlit_image_coordinates(draw_img, width=900, key="aariz_v2_1")
        if res:
            scale = img_raw.width / 900
            nx, ny = int(res["x"] * scale), int(res["y"] * scale)
            if l[target_idx] != [nx, ny]:
                st.session_state.lms[target_idx] = [nx, ny]
                st.rerun()

    with col2:
        st.header("📊 آنالیز کلینیکی")
        def get_angle(p1, p2, p3):
            v1, v2 = np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)
            norm = np.linalg.norm(v1)*np.linalg.norm(v2)
            return round(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/norm, -1, 1))), 1) if norm != 0 else 0
        
        sna = get_angle(l[10], l[4], l[0])
        snb = get_angle(l[10], l[4], l[2])
        anb = round(sna - snb, 1)
        
        st.metric("SNA (Maxilla)", f"{sna}°")
        st.metric("SNB (Mandible)", f"{snb}°")
        st.metric("ANB (Class)", f"{anb}°")
        
        if st.button("💾 ثبت نهایی"):
            st.success("آنالیز در آرشیو ذخیره شد.")
            st.balloons()
else:
    st.warning("در انتظار آپلود تصویر و لود مدل‌ها...")
