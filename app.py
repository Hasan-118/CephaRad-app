import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os, gdown, gc
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۱. اصلاح معماری بر اساس ساختار چک‌پوینت شما ---
# خطا نشان می‌دهد لایه ۴ در بلاک‌ها باید Conv باشد نه BatchNorm
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
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
        u1 = self.up1(x4); u1 = torch.cat([u1, x3], dim=1); c1 = self.conv_up1(u1)
        u2 = self.up2(c1); u2 = torch.cat([u2, x2], dim=1); c2 = self.conv_up2(u2)
        u3 = self.up3(c2); u3 = torch.cat([u3, x1], dim=1); c3 = self.conv_up3(u3)
        return self.outc(c3)

# --- ۲. لودر هوشمند با حذف کلیدهای اضافی ---
@st.cache_resource
def load_aariz_models():
    model_ids = {'m1': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks', 'm2': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU', 'm3': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'}
    dev = torch.device("cpu"); ms = []
    for k, fid in model_ids.items():
        path = f"{k}.pth"
        if not os.path.exists(path): gdown.download(f'https://drive.google.com/uc?id={fid}', path, quiet=True)
        m = CephaUNet().to(dev)
        ckpt = torch.load(path, map_location=dev, weights_only=False)
        sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        # تبدیل کلیدهای موازی (DataParallel) به معمولی
        new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
        # فقط لود کردن لایه‌هایی که ابعادشان در هر دو طرف یکی است
        m.load_state_dict({k: v for k, v in new_sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
        m.eval(); ms.append(m)
    return ms, dev

# --- ۳. توابع محاسباتی ---
def get_ang(p1, p2, p3, p4=None):
    v1 = np.array(p1)-np.array(p2) if p4 is None else np.array(p2)-np.array(p1)
    v2 = np.array(p3)-np.array(p2) if p4 is None else np.array(p4)-np.array(p3)
    norm = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    return round(float(np.degrees(np.arccos(np.clip(np.dot(v1, v2)/norm, -1, 1)))), 2)

# --- ۴. بدنه اصلی اپلیکیشن ---
st.set_page_config(page_title="Aariz Precision V7.8.7", layout="wide")
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']
models, device = load_aariz_models()

st.sidebar.title("🧬 Cepha Analysis")
pixel_size = st.sidebar.number_input("Pixel Size (mm):", 0.001, 1.0, 0.1, format="%.4f")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    if "lms" not in st.session_state or st.session_state.file_id != uploaded_file.name:
        img = Image.open(uploaded_file).convert("RGB")
        st.session_state.img = img
        # Inference
        W, H = img.size; ratio = 512 / max(W, H)
        img_rs = img.convert('L').resize((int(W*ratio), int(H*ratio)), Image.NEAREST)
        canvas = Image.new("L", (512, 512)); px, py = (512-img_rs.width)//2, (512-img_rs.height)//2
        canvas.paste(img_rs, (px, py)); tensor = transforms.ToTensor()(canvas).unsqueeze(0).to(device)
        with torch.no_grad():
            outs = [m(tensor)[0].cpu().numpy() for m in models]
            lms = {}
            for i in range(29):
                # انتخاب مدل بر اساس ناحیه (General, Anterior, TMJ)
                m_idx = 1 if i in {10, 14, 9, 5, 28, 20} else (2 if i in {7, 11, 12, 15} else 0)
                y, x = divmod(np.argmax(outs[m_idx][i]), 512)
                lms[i] = [int((x - px) / ratio), int((y - py) / ratio)]
        st.session_state.lms = lms
        st.session_state.file_id = uploaded_file.name
        st.session_state.v = 0

    l = st.session_state.lms; img = st.session_state.img; W, H = img.size
    
    # نمایش مگنیفایر و تصویر اصلی
    target_idx = st.sidebar.selectbox("🎯 Active Landmark:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")
    c1, c2 = st.columns([1, 2.5])
    
    with c1:
        st.subheader("Magnifier")
        cur = l[target_idx]; box = 80
        left, top = max(0, cur[0]-box), max(0, cur[1]-box)
        crop = img.crop((left, top, min(W, cur[0]+box), min(H, cur[1]+box))).resize((350, 350), Image.NEAREST)
        res_m = streamlit_image_coordinates(crop, key=f"m_{st.session_state.v}")
        if res_m:
            l[target_idx] = [int(left + (res_m['x'] * (2*box/350))), int(top + (res_m['y'] * (2*box/350)))]
            st.session_state.v += 1; st.rerun()

    with c2:
        sc = 800 / W; disp = img.resize((800, int(H*sc)), Image.NEAREST)
        draw = ImageDraw.Draw(disp)
        for i, p in l.items():
            clr = (255,0,0) if i == target_idx else (0,255,0)
            draw.ellipse([p[0]*sc-3, p[1]*sc-3, p[0]*sc+3, p[1]*sc+3], fill=clr)
        
        res_main = streamlit_image_coordinates(disp, width=800, key=f"main_{st.session_state.v}")
        if res_main:
            l[target_idx] = [int(res_main['x']/sc), int(res_main['y']/sc)]
            st.session_state.v += 1; st.rerun()

    # گزارش Steiner سریع
    with st.expander("📊 Steiner Quick Report"):
        sna = get_ang(l[10], l[4], l[0])
        snb = get_ang(l[10], l[4], l[2])
        st.write(f"SNA: {sna}° | SNB: {snb}° | ANB: {round(sna-snb, 2)}°")
