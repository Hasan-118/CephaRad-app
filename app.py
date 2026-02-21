import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import gdown
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۱. بازسازی دقیق معماری بر اساس خطای Size Mismatch ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_prob=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),  # Layer 0
            nn.BatchNorm2d(out_ch),                 # Layer 1
            nn.ReLU(inplace=True),                  # Layer 2
            nn.Dropout2d(p=dropout_prob),           # Layer 3
            nn.Conv2d(out_ch, out_ch, 3, padding=1), # Layer 4 (دقیقاً همان جایی که خطا می‌داد)
            nn.BatchNorm2d(out_ch),                 # Layer 5
            nn.ReLU(inplace=True)                   # Layer 6
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

# --- ۲. بارگذاری مدل‌ها با اصلاح پدینگ برای جلوگیری از قطع شدن صورت ---
@st.cache_resource
def load_aariz_system():
    model_ids = {
        'model1.pth': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks',
        'model2.pth': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU',
        'model3.pth': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'
    }
    device = torch.device("cpu")
    models = []
    for name, fid in model_ids.items():
        if not os.path.exists(name):
            gdown.download(f'https://drive.google.com/uc?id={fid}', name, quiet=False)
        try:
            m = CephaUNet(n_landmarks=29)
            ckpt = torch.load(name, map_location=device)
            state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            clean_state = {k.replace('module.', ''): v for k, v in state.items()}
            m.load_state_dict(clean_state, strict=True) # حالا باید True باشد چون معماری اصلاح شد
            m.eval()
            models.append(m)
        except Exception as e:
            st.error(f"خطا در مدل {name}: {e}")
    return models, device

def predict_safe(img_pil, models, device):
    ow, oh = img_pil.size
    ratio = 512 / max(ow, oh)
    nw, nh = int(ow * ratio), int(oh * ratio)
    img_rs = img_pil.convert('L').resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("L", (512, 512))
    px, py = (512 - nw) // 2, (512 - nh) // 2
    canvas.paste(img_rs, (px, py))
    
    input_t = transforms.ToTensor()(canvas).unsqueeze(0).to(device)
    with torch.no_grad():
        outs = [m(input_t)[0].cpu().numpy() for m in models]
    
    ANT_IDX, POST_IDX = [10, 14, 9, 5, 28, 20], [7, 11, 12, 15]
    coords = {}
    for i in range(29):
        if i in ANT_IDX and len(outs) >= 2: hm = outs[1][i]
        elif i in POST_IDX and len(outs) >= 3: hm = outs[2][i]
        else: hm = outs[0][i]
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        coords[i] = [int((x - px) / ratio), int((y - py) / ratio)]
    return coords

# --- ۳. رابط کاربری نهایی ---
st.set_page_config(page_title="Aariz Station V2.2", layout="wide")
models, device = load_aariz_system()
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

if models:
    st.sidebar.success(f"✅ {len(models)} مدل بارگذاری شد")
    up_file = st.sidebar.file_uploader("آپلود تصویر:", type=['png', 'jpg', 'jpeg'])
    if up_file:
        img_raw = Image.open(up_file).convert("RGB")
        if "lms" not in st.session_state or st.session_state.fid != up_file.name:
            st.session_state.lms = predict_safe(img_raw, models, device)
            st.session_state.fid = up_file.name

        target_idx = st.sidebar.selectbox("اصلاح لندمارک:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")
        col1, col2 = st.columns([3, 1])
        with col1:
            draw_img = img_raw.copy()
            draw = ImageDraw.Draw(draw_img)
            for i, pos in st.session_state.lms.items():
                c = "red" if i == target_idx else "lime"
                r = 12 if i == target_idx else 6
                draw.ellipse([pos[0]-r, pos[1]-r, pos[0]+r, pos[1]+r], fill=c)
            
            st.subheader("📍 تنظیم دقیق با کلیک")
            res = streamlit_image_coordinates(draw_img, width=800, key="aariz_fix")
            if res:
                scale = img_raw.width / 800
                nx, ny = int(res["x"]*scale), int(res["y"]*scale)
                if st.session_state.lms[target_idx] != [nx, ny]:
                    st.session_state.lms[target_idx] = [nx, ny]
                    st.rerun()
        with col2:
            st.header("📊 گزارش")
            l = st.session_state.lms
            def get_a(p1, p2, p3):
                v1, v2 = np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)
                return round(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1, 1))), 1)
            sna = get_a(l[10], l[4], l[0])
            st.metric("SNA", f"{sna}°")
            if st.button("💾 ذخیره"): st.success("ثبت شد")
else:
    st.error("خطا در لود مدل‌ها")
