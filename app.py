import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import gdown
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# [ساختار مدل CephaUNet و لودر مدل‌ها مطابق نسخه مرجع V3.3 - برای اختصار در اینجا تکرار نشده]

def get_precision_magnifier(img, coord, zoom=4, size=100):
    x, y = coord
    left, top = max(0, int(x - size//2)), max(0, int(y - size//2))
    right, bottom = min(img.width, int(x + size//2)), min(img.height, int(y + size//2))
    crop = img.crop((left, top, right, bottom)).resize((400, 400), Image.LANCZOS)
    
    draw_mag = ImageDraw.Draw(crop)
    cx, cy = 200, 200 
    draw_mag.line((cx-20, cy, cx+20, cy), fill="red", width=2)
    draw_mag.line((cx, cy-20, cx, cy+20), fill="red", width=2)
    return crop, (left, top) # برگرداندن مختصات گوشه برای محاسبه جابجایی

# --- رابط کاربری ---
st.set_page_config(page_title="Aariz Precision V3.4", layout="wide")
# [کد لود مدل و آپلود فایل...]

if uploaded_file:
    # [کد پیش‌بینی AI...]
    
    col1, col2 = st.columns([1.5, 2]) # تغییر چیدمان برای تمرکز بر ذره‌بین
    
    with col1:
        st.subheader("🎯 Micro-Adjustment")
        st.write("برای تنظیم میلی‌متری، روی هدف زیر کلیک کنید:")
        
        # نمایش ذره‌بین تعاملی
        mag_img, (offset_x, offset_y) = get_precision_magnifier(raw_img, st.session_state.lms[target_idx])
        
        # گرفتن کلیک از داخل ذره‌بین
        res_mag = streamlit_image_coordinates(mag_img, key="mag_click")
        
        if res_mag:
            # تبدیل مختصات کلیک در ذره‌بین (۴۰۰x۴۰۰) به مختصات تصویر اصلی
            click_x_in_crop = res_mag["x"] * (100 / 400) # بر اساس size=100 در تابع
            click_y_in_crop = res_mag["y"] * (100 / 400)
            
            new_x = int(offset_x + click_x_in_crop)
            new_y = int(offset_y + click_y_in_crop)
            
            if st.session_state.lms[target_idx] != [new_x, new_y]:
                st.session_state.lms[target_idx] = [new_x, new_y]
                st.rerun()

    with col2:
        st.subheader("🖼 Full View")
        # [رسم تصویر اصلی و گرفتن کلیک برای جابجایی‌های بزرگ...]
        # (دقیقا مشابه نسخه قبل)import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import gdown
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۱. ساختار مدل مرجع (ثابت طبق حافظه) ---
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

# --- ۲. لودر مدل و مگنیفایر پیشرفته ---
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

def get_precision_magnifier(img, coord, zoom=4, size=120):
    x, y = coord
    left, top = max(0, int(x - size//2)), max(0, int(y - size//2))
    right, bottom = min(img.width, int(x + size//2)), min(img.height, int(y + size//2))
    
    # برش و زوم
    crop = img.crop((left, top, right, bottom)).resize((400, 400), Image.LANCZOS)
    
    # رسم نشانگر مرکز (Crosshair) روی مگنیفایر
    draw_mag = ImageDraw.Draw(crop)
    cx, cy = 200, 200 # مرکز تصویر ۴۰۰ پیکسلی
    draw_mag.line((cx-20, cy, cx+20, cy), fill="red", width=2)
    draw_mag.line((cx, cy-20, cx, cy+20), fill="red", width=2)
    draw_mag.ellipse([cx-4, cy-4, cx+4, cy+4], outline="white", width=1)
    return crop

# --- ۳. رابط کاربری ---
st.set_page_config(page_title="Aariz AI V3.3", layout="wide")
models, device = load_aariz_system()
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

uploaded_file = st.sidebar.file_uploader("آپلود تصویر:", type=['png', 'jpg', 'jpeg'])

if uploaded_file and models:
    raw_img = Image.open(uploaded_file).convert("RGB")
    
    if "lms" not in st.session_state or st.session_state.get("file_id") != uploaded_file.name:
        # Prediction Logic...
        img_gray = raw_img.convert('L').resize((512, 512), Image.LANCZOS)
        t = transforms.ToTensor()(img_gray).unsqueeze(0).to(device)
        with torch.no_grad():
            outs = [m(t)[0].cpu().numpy() for m in models]
        coords = {}
        sx, sy = raw_img.width/512, raw_img.height/512
        for i in range(29):
            hm = outs[0][i]
            y, x = np.unravel_index(np.argmax(hm), hm.shape)
            coords[i] = [int(x * sx), int(y * sy)]
        st.session_state.lms = coords
        st.session_state.file_id = uploaded_file.name

    target_idx = st.sidebar.selectbox("🎯 انتخاب لندمارک فعال:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")

    col1, col2 = st.columns([2.5, 1])
    with col1:
        st.subheader(f"🔍 زوم دقیق: {landmark_names[target_idx]}")
        # نمایش مگنیفایر با علامت + در مرکز
        mag = get_precision_magnifier(raw_img, st.session_state.lms[target_idx])
        st.image(mag, width=350)

        # رسم تصویر اصلی
        draw_img = raw_img.copy()
        draw = ImageDraw.Draw(draw_img)
        l = st.session_state.lms
        for i, pos in l.items():
            color = "red" if i == target_idx else "#00FF00"
            r = 15 if i == target_idx else 8
            draw.ellipse([pos[0]-r, pos[1]-r, pos[0]+r, pos[1]+r], fill=color, outline="white", width=2)

        res = streamlit_image_coordinates(draw_img, width=900, key="aariz_v3_3")
        if res:
            scale = raw_img.width / 900
            new_coord = [int(res["x"]*scale), int(res["y"]*scale)]
            if st.session_state.lms[target_idx] != new_coord:
                st.session_state.lms[target_idx] = new_coord
                st.rerun()

    with col2:
        st.header("📊 آنالیز")
        # [محاسبات SNA/SNB مشابه قبل...]
        st.info("نشانگر قرمز در ذره‌بین، مرکز دقیق لندمارک انتخابی شماست.")

