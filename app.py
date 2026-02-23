import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os, gdown, gc, json
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۱. معماری مدل (Gold Standard) ---
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

# --- ۲. لودر هوشمند مدل‌ها ---
@st.cache_resource
def load_aariz_models():
    model_ids = {
        'checkpoint_unet_clinical.pth': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks', 
        'specialist_pure_model.pth': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU', 
        'tmj_specialist_model.pth': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'
    }
    dev = torch.device("cpu"); ms = []
    for f, fid in model_ids.items():
        if not os.path.exists(f): gdown.download(f'https://drive.google.com/uc?id={fid}', f, quiet=True)
        m = CephaUNet(n_landmarks=29).to(dev)
        ckpt = torch.load(f, map_location=dev, weights_only=False)
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        m.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=False)
        m.eval(); ms.append(m)
    gc.collect(); return ms, dev

def predict_fast(img_pil, ms, dev):
    W, H = img_pil.size; ratio = 512 / max(W, H)
    img_rs = img_pil.convert('L').resize((int(W*ratio), int(H*ratio)), Image.NEAREST)
    canvas = Image.new("L", (512, 512)); px, py = (512-img_rs.width)//2, (512-img_rs.height)//2
    canvas.paste(img_rs, (px, py)); tensor = transforms.ToTensor()(canvas).unsqueeze(0).to(dev)
    ANT_IDX, POST_IDX = {10, 14, 9, 5, 28, 20}, {7, 11, 12, 15}
    res = {}
    with torch.no_grad():
        outs = [m(tensor)[0].cpu().numpy() for m in ms]
        for i in range(29):
            m_idx = 1 if i in ANT_IDX else (2 if i in POST_IDX else 0)
            y, x = divmod(np.argmax(outs[m_idx][i]), 512)
            res[i] = [int((x - px) / ratio), int((y - py) / ratio)]
    return res

# --- ۳. توابع محاسباتی اصلاح شده (Fix Errors) ---
def get_ang(p1, p2, p3, p4=None):
    """محاسبه زاویه: اگر ۴ نقطه داده شود زاویه بین دو خط، در غیر این صورت زاویه بین ۳ نقطه با راس p2"""
    try:
        if p4 is None: # زاویه بین ۳ نقطه (مثلاً SNA) - راس p2
            v1 = np.array(p1) - np.array(p2)
            v2 = np.array(p3) - np.array(p2)
        else: # زاویه بین دو خط مستقل (مثلاً FH vs MP)
            v1 = np.array(p2) - np.array(p1)
            v2 = np.array(p4) - np.array(p3)
        
        norm = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-7
        return round(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / norm, -1, 1))), 2)
    except: return 0.0

def dist_to_line(p, l1, l2):
    """رفع هشدار NumPy 2.0 برای ضرب خارجی ۲ بعدی"""
    p, l1, l2 = np.array(p), np.array(l1), np.array(l2)
    # استفاده از فرمول قدر مطلق برای بردار ۲ بعدی جهت رفع DeprecationWarning
    return np.abs((l2[0]-l1[0])*(l1[1]-p[1]) - (l1[0]-p[0])*(l2[1]-l1[1])) / (np.linalg.norm(l2-l1) + 1e-6)

# --- ۴. رابط کاربری و نمایش ---
st.set_page_config(page_title="Aariz Precision Station V7.8.2", layout="wide")
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']
models, device = load_aariz_models()

st.sidebar.title("📏 تنظیمات کلینیکی")
gender = st.sidebar.radio("جنسیت:", ["آقا (Male)", "خانم (Female)"])
pixel_size = st.sidebar.number_input("Pixel Size (mm):", 0.001, 1.0, 0.1, format="%.4f")
analysis_mode = st.sidebar.selectbox("📊 نمایش آنالیز:", 
    ["فقط لندمارک‌ها", "Steiner (SNA/SNB)", "McNamara & FH", "Soft Tissue (E-Line)", "نمایش جامع"])
target_idx = st.sidebar.selectbox("🎯 لندمارک فعال:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")

uploaded_file = st.sidebar.file_uploader("آپلود تصویر", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    if "lms" not in st.session_state or st.session_state.file_id != uploaded_file.name:
        img = Image.open(uploaded_file).convert("RGB")
        st.session_state.img = img
        st.session_state.lms = predict_fast(img, models, device)
        st.session_state.file_id = uploaded_file.name
        st.session_state.v = 0

    l = st.session_state.lms
    img = st.session_state.img; W, H = img.size
    c1, c2 = st.columns([1.2, 2.8])

    with c1:
        st.subheader("🔍 مگنیفایر")
        cur = l[target_idx]; box = 100
        left, top = max(0, cur[0]-box), max(0, cur[1]-box)
        crop = img.crop((left, top, min(W, cur[0]+box), min(H, cur[1]+box))).resize((400, 400), Image.NEAREST)
        draw_m = ImageDraw.Draw(crop)
        draw_m.line((195, 200, 205, 200), fill="red", width=2); draw_m.line((200, 195, 200, 205), fill="red", width=2)
        res_m = streamlit_image_coordinates(crop, key=f"m_{target_idx}_{st.session_state.v}")
        if res_m:
            new_c = [int(left + (res_m['x'] * (2*box/400))), int(top + (res_m['y'] * (2*box/400)))]
            if new_c != l[target_idx]:
                l[target_idx] = new_c; st.session_state.v += 1; st.rerun()

    with c2:
        st.subheader("🖼 ترسیمات آنالیز")
        sc = 850 / W; disp = img.resize((850, int(H*sc)), Image.NEAREST)
        draw = ImageDraw.Draw(disp)
        def sp(idx): return (l[idx][0]*sc, l[idx][1]*sc)

        # ترسیم خطوط بر اساس مود انتخابی
        if analysis_mode != "فقط لندمارک‌ها":
            if "Steiner" in analysis_mode or "جامع" in analysis_mode:
                if all(k in l for k in [10, 4, 0, 2]):
                    draw.line([sp(10), sp(4)], fill="yellow", width=2)
                    draw.line([sp(4), sp(0)], fill="cyan", width=1)
                    draw.line([sp(4), sp(2)], fill="magenta", width=1)
            if "McNamara" in analysis_mode or "جامع" in analysis_mode:
                if all(k in l for k in [15, 5, 14, 3]):
                    draw.line([sp(15), sp(5)], fill="orange", width=2)
                    draw.line([sp(14), sp(3)], fill="purple", width=2)

        for i, p in l.items():
            clr = (255,0,0) if i == target_idx else (0,255,0)
            draw.ellipse([p[0]*sc-4, p[1]*sc-4, p[0]*sc+4, p[1]*sc+4], fill=clr)
            draw.text((p[0]*sc+5, p[1]*sc-5), landmark_names[i], fill=clr)

        res_main = streamlit_image_coordinates(disp, width=850, key=f"main_{st.session_state.v}")
        if res_main:
            l[target_idx] = [int(res_main['x']/sc), int(res_main['y']/sc)]
            st.session_state.v += 1; st.rerun()

    # --- ۵. گزارش نهایی (بدون خطا) ---
    with st.expander("📊 مشاهده گزارش تحلیلی"):
        sna, snb = get_ang(l[10], l[4], l[0]), get_ang(l[10], l[4], l[2])
        anb = round(sna - snb, 2)
        fma = get_ang(l[15], l[5], l[14], l[3]) # ارسال ۴ نقطه بدون خطا
        co_a = np.linalg.norm(np.array(l[12])-np.array(l[0])) * pixel_size
        co_gn = np.linalg.norm(np.array(l[12])-np.array(l[13])) * pixel_size
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Steiner (ANB)", f"{anb}°", f"SNA: {sna}°")
        m2.metric("McNamara Diff", f"{round(co_gn-co_a, 1)} mm")
        m3.metric("FMA Angle", f"{fma}°")
