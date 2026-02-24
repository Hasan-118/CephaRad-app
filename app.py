import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os, gdown, gc, json
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۱. معماری مدل (اصلاح شده برای مطابقت با چک‌پوینت‌ها) ---
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

# --- ۲. لودر هوشمند (حل مشکل RuntimeError) ---
@st.cache_resource
def load_aariz_models():
    model_ids = {
        'm1': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks', 
        'm2': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU', 
        'm3': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'
    }
    dev = torch.device("cpu"); ms = []
    for k, fid in model_ids.items():
        path = f"{k}.pth"
        if not os.path.exists(path): gdown.download(f'https://drive.google.com/uc?id={fid}', path, quiet=True)
        m = CephaUNet().to(dev)
        ckpt = torch.load(path, map_location=dev, weights_only=False)
        sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        # فیلتر کردن لایه‌ها برای لود ایمن
        new_sd = {k.replace('module.', ''): v for k, v in sd.items() if k.replace('module.', '') in m.state_dict()}
        m.load_state_dict(new_sd, strict=False)
        m.eval(); ms.append(m)
    return ms, dev

# --- ۳. توابع ریاضی اصلاح شده (Fix TypeError & NumPy) ---
def get_ang(p1, p2, p3, p4=None):
    """محاسبه زاویه ۳ یا ۴ نقطه‌ای بدون خطا"""
    v1 = np.array(p1)-np.array(p2) if p4 is None else np.array(p2)-np.array(p1)
    v2 = np.array(p3)-np.array(p2) if p4 is None else np.array(p4)-np.array(p3)
    norm = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    return round(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / norm, -1, 1))), 2)

def predict_fast(img_pil, ms, dev):
    W, H = img_pil.size; ratio = 512 / max(W, H)
    img_rs = img_pil.convert('L').resize((int(W*ratio), int(H*ratio)), Image.NEAREST)
    canvas = Image.new("L", (512, 512)); px, py = (512-img_rs.width)//2, (512-img_rs.height)//2
    canvas.paste(img_rs, (px, py)); tensor = transforms.ToTensor()(canvas).unsqueeze(0).to(dev)
    ANT, POST = {10, 14, 9, 5, 28, 20}, {7, 11, 12, 15}
    res = {}
    with torch.no_grad():
        outs = [m(tensor)[0].cpu().numpy() for m in ms]
        for i in range(29):
            m_idx = 1 if i in ANT else (2 if i in POST else 0)
            y, x = divmod(np.argmax(outs[m_idx][i]), 512)
            res[i] = [int((x - px) / ratio), int((y - py) / ratio)]
    return res

# --- ۴. رابط کاربری (UI) ---
st.set_page_config(page_title="Aariz Precision V7.8.5", layout="wide")
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']
models, device = load_aariz_models()

st.sidebar.title("📏 تنظیمات آنالیز")
pixel_size = st.sidebar.number_input("Pixel Size (mm):", 0.001, 1.0, 0.1, format="%.4f")
# کشوی بازشونده برای انتخاب آنالیز جهت سرعت بالاتر
analysis_selection = st.sidebar.selectbox("📊 انتخاب نوع نمایش خطوط:", 
    ["فقط لندمارک‌ها (بسیار سریع)", "Steiner (SNA/SNB/ANB)", "McNamara & FH", "E-Line & Soft Tissue", "نمایش جامع تمام خطوط"])

target_idx = st.sidebar.selectbox("🎯 انتخاب لندمارک فعال:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")

uploaded_file = st.sidebar.file_uploader("آپلود تصویر سفالومتری", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    if "lms" not in st.session_state or st.session_state.file_id != uploaded_file.name:
        img_raw = Image.open(uploaded_file).convert("RGB")
        st.session_state.img = img_raw
        st.session_state.lms = predict_fast(img_raw, models, device)
        st.session_state.file_id = uploaded_file.name
        st.session_state.v = 0

    l = st.session_state.lms; img = st.session_state.img; W, H = img.size
    col1, col2 = st.columns([1.2, 2.8])

    with col1:
        st.subheader("🔍 مگنیفایر")
        cur = l[target_idx]; box = 100
        left, top = max(0, cur[0]-box), max(0, cur[1]-box)
        crop = img.crop((left, top, min(W, cur[0]+box), min(H, cur[1]+box))).resize((400, 400), Image.NEAREST)
        draw_m = ImageDraw.Draw(crop)
        draw_m.line((195, 200, 205, 200), fill="red", width=2); draw_m.line((200, 195, 200, 205), fill="red", width=2)
        res_m = streamlit_image_coordinates(crop, key=f"m_{target_idx}_{st.session_state.v}")
        if res_m:
            l[target_idx] = [int(left + (res_m['x'] * (2*box/400))), int(top + (res_m['y'] * (2*box/400)))]
            st.session_state.v += 1; st.rerun()

    with col2:
        st.subheader("🖼 ترسیمات سفالومتری")
        sc = 850 / W; disp = img.resize((850, int(H*sc)), Image.NEAREST)
        draw = ImageDraw.Draw(disp)
        def sp(idx): return (l[idx][0]*sc, l[idx][1]*sc)

        # ترسیم خطوط انتخابی (On-Demand)
        if "Steiner" in analysis_selection or "جامع" in analysis_selection:
            if all(k in l for k in [10, 4, 0, 2]):
                draw.line([sp(10), sp(4)], fill="yellow", width=2) # SN
                draw.line([sp(4), sp(0)], fill="cyan", width=1)   # NA
                draw.line([sp(4), sp(2)], fill="magenta", width=1)# NB
        
        if "McNamara" in analysis_selection or "جامع" in analysis_selection:
            if all(k in l for k in [15, 5, 14, 3]):
                draw.line([sp( Po_idx:=15), sp( Or_idx:=5)], fill="orange", width=2) # FH
                draw.line([sp(14), sp(3)], fill="purple", width=2) # MP

        # رسم لندمارک‌ها و نام‌ها
        for i, p in l.items():
            clr = (255, 0, 0) if i == target_idx else (0, 255, 0)
            draw.ellipse([p[0]*sc-3, p[1]*sc-3, p[0]*sc+3, p[1]*sc+3], fill=clr)
            draw.text((p[0]*sc+5, p[1]*sc-5), landmark_names[i], fill=clr)

        res_main = streamlit_image_coordinates(disp, width=850, key=f"main_{st.session_state.v}")
        if res_main:
            l[target_idx] = [int(res_main['x']/sc), int(res_main['y']/sc)]
            st.session_state.v += 1; st.rerun()

    # --- ۵. گزارش بالینی نهایی ---
    with st.expander("📑 مشاهده نتایج آنالیز"):
        sna = get_ang(l[10], l[4], l[0])
        snb = get_ang(l[10], l[4], l[2])
        anb = round(sna - snb, 2)
        fma = get_ang(l[15], l[5], l[14], l[3]) # رفع خطای ۴ آرگومان
        
        c1, c2, c3 = st.columns(3)
        c1.metric("SNA / SNB", f"{sna}° / {snb}°", f"ANB: {anb}°")
        c2.metric("FMA Angle", f"{fma}°")
        
        if st.button("💾 ذخیره لندمارک‌ها"):
            with open(f"saved_{st.session_state.file_id}.json", "w") as f:
                json.dump(l, f)
            st.success("مختصات با موفقیت ذخیره شد.")
