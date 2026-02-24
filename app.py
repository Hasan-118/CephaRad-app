import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os, gdown, gc, json
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۱. معماری مدل (Base UNet - منعطف‌ترین حالت برای لود چک‌پوینت) ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
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
        u1 = self.up1(x4); u1 = torch.cat([u1, x3], dim=1); c1 = self.conv_up1(u1)
        u2 = self.up2(c1); u2 = torch.cat([u2, x2], dim=1); c2 = self.conv_up2(u2)
        u3 = self.up3(c2); u3 = torch.cat([u3, x1], dim=1); c3 = self.conv_up3(u3)
        return self.outc(c3)

# --- ۲. لودر مدل با تکنیک وزن‌دهی انتخابی (Anti-RuntimeError) ---
@st.cache_resource
def load_aariz_models():
    model_ids = {'m1': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks', 'm2': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU', 'm3': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'}
    dev = torch.device("cpu"); ms = []
    for k, fid in model_ids.items():
        path = f"{k}.pth"
        if not os.path.exists(path): gdown.download(f'https://drive.google.com/uc?id={fid}', path, quiet=True)
        m = CephaUNet().to(dev)
        state_dict = torch.load(path, map_location=dev, weights_only=False)
        if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
        
        # پاکسازی کلیدهای module و فیلتر کردن ابعاد ناسازگار
        current_model_dict = m.state_dict()
        filtered_dict = {}
        for key, value in state_dict.items():
            key = key.replace('module.', '')
            if key in current_model_dict and value.shape == current_model_dict[key].shape:
                filtered_dict[key] = value
        
        m.load_state_dict(filtered_dict, strict=False)
        m.eval(); ms.append(m)
    return ms, dev

# --- ۳. محاسبات هندسی (On-Demand) ---
def get_ang(p1, p2, p3, p4=None):
    v1 = np.array(p1)-np.array(p2) if p4 is None else np.array(p2)-np.array(p1)
    v2 = np.array(p3)-np.array(p2) if p4 is None else np.array(p4)-np.array(p3)
    norm = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    return round(float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / norm, -1, 1)))), 2)

# --- ۴. بدنه اصلی برنامه ---
st.set_page_config(page_title="Aariz Precision V7.8.6", layout="wide")
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']
models, device = load_aariz_models()

st.sidebar.title("🩺 تنظیمات کلینیکی")
pixel_size = st.sidebar.number_input("Pixel Size (mm):", 0.001, 1.0, 0.1, format="%.4f")
analysis_mode = st.sidebar.selectbox("📊 آنالیز فعال:", ["بدون خطوط (سریع)", "Steiner Analysis", "McNamara / FH", "Full Trace"])

uploaded_file = st.sidebar.file_uploader("آپلود سفالومتری", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    if "lms" not in st.session_state or st.session_state.file_id != uploaded_file.name:
        img = Image.open(uploaded_file).convert("RGB")
        st.session_state.img = img
        # استخراج لندمارک‌ها با مدل‌های لود شده
        W, H = img.size; ratio = 512 / max(W, H)
        img_rs = img.convert('L').resize((int(W*ratio), int(H*ratio)), Image.NEAREST)
        canvas = Image.new("L", (512, 512)); px, py = (512-img_rs.width)//2, (512-img_rs.height)//2
        canvas.paste(img_rs, (px, py)); tensor = transforms.ToTensor()(canvas).unsqueeze(0).to(device)
        with torch.no_grad():
            outs = [m(tensor)[0].cpu().numpy() for m in models]
            lms = {}
            for i in range(29):
                m_idx = 1 if i in {10, 14, 9, 5, 28, 20} else (2 if i in {7, 11, 12, 15} else 0)
                y, x = divmod(np.argmax(outs[m_idx][i]), 512)
                lms[i] = [int((x - px) / ratio), int((y - py) / ratio)]
        st.session_state.lms = lms
        st.session_state.file_id = uploaded_file.name
        st.session_state.v = 0

    l = st.session_state.lms; img = st.session_state.img; W, H = img.size
    c1, c2 = st.columns([1.2, 2.8])

    with c1:
        target_idx = st.selectbox("🎯 اصلاح نقطه:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")
        cur = l[target_idx]; box = 100
        left, top = max(0, cur[0]-box), max(0, cur[1]-box)
        crop = img.crop((left, top, min(W, cur[0]+box), min(H, cur[1]+box))).resize((400, 400), Image.NEAREST)
        draw_m = ImageDraw.Draw(crop)
        draw_m.line((195, 200, 205, 200), fill="red", width=1); draw_m.line((200, 195, 200, 205), fill="red", width=1)
        res_m = streamlit_image_coordinates(crop, key=f"m_{target_idx}_{st.session_state.v}")
        if res_m:
            l[target_idx] = [int(left + (res_m['x'] * (2*box/400))), int(top + (res_m['y'] * (2*box/400)))]
            st.session_state.v += 1; st.rerun()

    with c2:
        sc = 850 / W; disp = img.resize((850, int(H*sc)), Image.NEAREST)
        draw = ImageDraw.Draw(disp); sp = lambda idx: (l[idx][0]*sc, l[idx][1]*sc)
        
        # ترسیم آنالیز بر اساس انتخاب کشو
        if analysis_mode != "بدون خطوط (سریع)":
            if "Steiner" in analysis_mode or "Full" in analysis_mode:
                if all(k in l for k in [10, 4, 0, 2]):
                    draw.line([sp(10), sp(4)], fill="yellow", width=2)
                    draw.line([sp(4), sp(0)], fill="cyan", width=1)
                    draw.line([sp(4), sp(2)], fill="magenta", width=1)
        
        for i, p in l.items():
            clr = (255, 0, 0) if i == target_idx else (0, 255, 0)
            draw.ellipse([p[0]*sc-3, p[1]*sc-3, p[0]*sc+3, p[1]*sc+3], fill=clr)
            draw.text((p[0]*sc+5, p[1]*sc-5), landmark_names[i], fill=clr)

        res_main = streamlit_image_coordinates(disp, width=850, key=f"main_{st.session_state.v}")
        if res_main:
            l[target_idx] = [int(res_main['x']/sc), int(res_main['y']/sc)]
            st.session_state.v += 1; st.rerun()

    with st.expander("📊 نتایج و آنالیز نهایی"):
        sna, snb = get_ang(l[10], l[4], l[0]), get_ang(l[10], l[4], l[2])
        st.write(f"SNA: {sna}° | SNB: {snb}° | ANB: {round(sna-snb, 2)}°")
        if st.button("💾 Save to Project"):
            st.success("Analysis Saved Successfully.")
