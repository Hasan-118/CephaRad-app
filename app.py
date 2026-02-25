import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import gdown
import gc
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۱. معماری مرجع Aariz (بدون تغییر طبق دستور) ---
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

# --- ۲. بارگذاری مدل‌ها و پیش‌بینی هوشمند (با تفکیک مناطق متخصص) ---
@st.cache_resource
def load_aariz_models():
    model_ids = {
        'checkpoint_unet_clinical.pth': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks', 
        'specialist_pure_model.pth': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU', 
        'tmj_specialist_model.pth': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'
    }
    device = torch.device("cpu"); loaded_models = []
    for f, fid in model_ids.items():
        if not os.path.exists(f): gdown.download(f'https://drive.google.com/uc?id={fid}', f, quiet=True)
        try:
            m = CephaUNet(n_landmarks=29).to(device)
            ckpt = torch.load(f, map_location=device)
            state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            m.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=False)
            m.eval(); loaded_models.append(m)
        except: pass
    return loaded_models, device

def run_precise_prediction(img_pil, models, device):
    ow, oh = img_pil.size; img_gray = img_pil.convert('L'); ratio = 512 / max(ow, oh)
    nw, nh = int(ow * ratio), int(oh * ratio); img_rs = img_gray.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("L", (512, 512)); px, py = (512 - nw) // 2, (512 - nh) // 2
    canvas.paste(img_rs, (px, py)); input_tensor = transforms.ToTensor()(canvas).unsqueeze(0).to(device)
    with torch.no_grad(): outs = [m(input_tensor)[0].cpu().numpy() for m in models]
    
    # نقاط تخصصی آموزش دیده در مدل‌های دوم و سوم
    ANT_IDX = [10, 14, 9, 5, 28, 20] 
    POST_IDX = [7, 11, 12, 15]
    
    coords = {}
    for i in range(29):
        # انتخاب مدل بر اساس ناحیه آناتومیک
        heatmap = outs[1][i] if i in ANT_IDX else (outs[2][i] if i in POST_IDX else outs[0][i])
        idx = np.unravel_index(np.argmax(heatmap), (512, 512))
        coords[i] = [int((idx[1] - px) / ratio), int((idx[0] - py) / ratio)]
    
    gc.collect(); return coords

# --- ۳. رابط کاربری (UI) و مدیریت وضعیت ---
st.set_page_config(page_title="Aariz Precision Station V7.8.23", layout="wide")
models, device = load_aariz_models()
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

if "click_version" not in st.session_state: st.session_state.click_version = 0

st.sidebar.header("📏 تنظیمات بیمار")
gender = st.sidebar.radio("جنسیت بیمار:", ["آقا (Male)", "خانم (Female)"])
pixel_size = st.sidebar.number_input("Pixel Size (mm/px):", 0.01, 1.0, 0.1, 0.0001, format="%.4f")
text_scale = st.sidebar.slider("🔤 مقیاس نام لندمارک:", 1, 10, 3)

uploaded_file = st.sidebar.file_uploader("آپلود تصویر سفالومتری:", type=['png', 'jpg', 'jpeg'])

if uploaded_file and len(models) == 3:
    raw_img = Image.open(uploaded_file).convert("RGB"); W, H = raw_img.size
    if "lms" not in st.session_state or st.session_state.get("file_id") != uploaded_file.name:
        st.session_state.lms = run_precise_prediction(raw_img, models, device)
        st.session_state.file_id = uploaded_file.name

    target_idx = st.sidebar.selectbox("🎯 انتخاب لندمارک فعال:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")
    
    col1, col2 = st.columns([1.2, 2.5])
    with col1:
        st.subheader("🔍 Micro-Adjustment")
        l_pos = st.session_state.lms[target_idx]; sz_m = 180 
        left, top = max(0, min(int(l_pos[0]-sz_m//2), W-sz_m)), max(0, min(int(l_pos[1]-sz_m//2), H-sz_m))
        mag_crop = raw_img.crop((left, top, left+sz_m, top+sz_m)).resize((400, 400), Image.LANCZOS)
        mag_draw = ImageDraw.Draw(mag_crop)
        # نشانه مرکزی برای تنظیم دقیق
        mag_draw.line((190, 200, 210, 200), fill="red", width=2)
        mag_draw.line((200, 190, 200, 210), fill="red", width=2)
        
        res_mag = streamlit_image_coordinates(mag_crop, key=f"mag_{st.session_state.click_version}")
        if res_mag and isinstance(res_mag, dict):
            new_c = [int(left + (res_mag["x"] * sz_m / 400)), int(top + (res_mag["y"] * sz_m / 400))]
            if st.session_state.lms[target_idx] != new_c:
                st.session_state.lms[target_idx] = new_c; st.session_state.click_version += 1; st.rerun()

    with col2:
        st.subheader("🖼 نمای گرافیکی و خطوط آنالیز")
        draw_img = raw_img.copy(); draw = ImageDraw.Draw(draw_img); l = st.session_state.lms
        
        # ترسیم خطوط آنالیز بالینی
        analysis_lines = [(10,4,"yellow"), (4,0,"cyan"), (4,2,"magenta"), (15,5,"orange"), (14,3,"purple"), (8,27,"pink"), (12,0,"red"), (12,13,"lime")]
        for p1, p2, clr in analysis_lines:
            if p1 in l and p2 in l: draw.line([tuple(l[p1]), tuple(l[p2])], fill=clr, width=3)

        for i, pos in l.items():
            color = (255, 0, 0) if i == target_idx else (0, 255, 0)
            r = 10 if i == target_idx else 6
            draw.ellipse([pos[0]-r, pos[1]-r, pos[0]+r, pos[1]+r], fill=color, outline="white", width=2)
            # نمایش نام لندمارک با ابعاد داینامیک
            draw.text((pos[0]+r+5, pos[1]-r), landmark_names[i], fill=color)

        res_main = streamlit_image_coordinates(draw_img, width=850, key=f"main_{st.session_state.click_version}")
        if res_main and isinstance(res_main, dict):
            m_c = [int(res_main["x"] * W / 850), int(res_main["y"] * H / 850)]
            if st.session_state.lms[target_idx] != m_c:
                st.session_state.lms[target_idx] = m_c; st.session_state.click_version += 1; st.rerun()

    # --- ۴. محاسبات پیشرفته و گزارش نهایی ---
    st.divider()
    def get_ang(p1, p2, p3, p4=None):
        """محاسبه زاویه بین ۳ نقطه یا بین دو خط (۴ نقطه)"""
        if p4 is None:
            v1, v2 = np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)
        else:
            v1, v2 = np.array(p2)-np.array(p1), np.array(p4)-np.array(p3)
        n = np.linalg.norm(v1)*np.linalg.norm(v2)
        return round(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n if n>0 else 1e-6), -1, 1))), 2)

    def dist_to_line(p, l1, l2):
        """محاسبه فاصله یک نقطه از یک خط (برای آنالیزهای عمودی)"""
        p3, l1_3, l2_3 = np.append(p, 0), np.append(l1, 0), np.append(l2, 0)
        return round(np.linalg.norm(np.cross(l2_3-l1_3, l1_3-p3)) / (np.linalg.norm(l2_3-l1_3) + 1e-6) * pixel_size, 2)

    l = st.session_state.lms
    sna, snb = get_ang(l[10], l[4], l[0]), get_ang(l[10], l[4], l[2])
    anb = round(sna - snb, 2)
    fma = get_ang(l[15], l[5], l[14], l[3])
    # آنالیز مک‌نامارا (تفاضل طول فک بالا و پایین)
    diff_mcnamara = round((np.linalg.norm(np.array(l[12])-np.array(l[13])) - np.linalg.norm(np.array(l[12])-np.array(l[0]))) * pixel_size, 2)
    
    st.header(f"📑 گزارش تحلیلی Aariz")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🦷 وضعیت فکی (Skeletal)")
        st.metric("ANB Angle", f"{anb}°", f"SNA: {sna} / SNB: {snb}")
        st.metric("McNamara Diff", f"{diff_mcnamara} mm", "Target: 25-30mm")
        diag = "Class II" if anb > 4 else "Class III" if anb < 0 else "Class I"
        st.info(f"**Diagnosis:** {diag}")

    with c2:
        st.subheader("👄 بافت نرم و رشد")
        fma_desc = "Vertical" if fma > 30 else "Horizontal" if fma < 20 else "Normal"
        st.warning(f"**Growth Pattern:** {fma_desc} ({fma}°)")
        
        if st.button("📄 تولید گزارش نهایی و کد آنالیز"):
            st.success("گزارش با موفقیت ایجاد شد.")
            st.code(f"Aariz Precise Report\n---\nANB: {anb}\nFMA: {fma}\nMcNamara: {diff_mcnamara}mm\nPattern: {fma_desc}")

    gc.collect()
