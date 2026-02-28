import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import gc
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۱. تنظیمات صفحه و استایل (فشرده‌سازی) ---
st.set_page_config(page_title="Aariz Precision Station V7.8.48", layout="wide")

# اعمال CSS برای کاهش سایز کلی عناصر
st.markdown("""
<style>
    /* کاهش سایز فونت کلی */
    html, body, [class*="css"]  {
        font-size: 14px;
    }
    /* کاهش فاصله عناصر */
    .stButton>button { padding: 0.2rem 0.5rem; font-size: 12px; }
    .stSelectbox, .stRadio, .stNumberInput, .stFileUploader { margin-top: -10px; }
    /* کاهش سایز سایدبار */
    [data-testid="stSidebar"] { min-width: 250px; max-width: 300px; }
</style>
""", unsafe_allow_html=True)

# --- ۲. معماری مرجع (بدون تغییر) ---
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

# --- ۳. مدیریت فایل و مدل‌ها ---
def get_model_map():
    return ['checkpoint_unet_clinical.pth', 'specialist_pure_model.pth', 'tmj_specialist_model.pth']

def check_files():
    for f in get_model_map():
        if not os.path.exists(f):
            st.error(f"❌ فایل `{f}` پیدا نشد.")
            return False
    return True

@st.cache_resource
def load_models():
    if not check_files(): return None
    device = torch.device("cpu")
    loaded_models = []
    for f in get_model_map():
        m = CephaUNet(n_landmarks=29).to(device)
        ckpt = torch.load(f, map_location=device)
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        m.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=False)
        m.eval()
        loaded_models.append(m)
    return loaded_models

# --- ۴. اجرای منطق اصلی برنامه ---
models = load_models()

st.sidebar.title("🛠 مرکز پردازش Aariz")
gender = st.sidebar.radio("جنسیت بیمار:", ["آقا (Male)", "خانم (Female)"])
pixel_size = st.sidebar.number_input("Pixel Size (mm/px):", 0.01, 1.0, 0.1, 0.001, format="%.4f")
text_scale = st.sidebar.slider("🔤 مقیاس نام لندمارک:", 1, 10, 2) # پیش‌فرض کمتر شد
uploaded_file = st.sidebar.file_uploader("آپلود تصویر:", type=['png', 'jpg', 'jpeg'])

if models is None:
    st.stop()

# --- ۵. پردازش تصویر و تنظیم ابعاد (فشرده‌تر) ---
def run_precise_prediction(img_pil, models):
    device = torch.device("cpu")
    ow, oh = img_pil.size; img_gray = img_pil.convert('L'); ratio = 512 / max(ow, oh)
    nw, nh = int(ow * ratio), int(oh * ratio); img_rs = img_gray.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("L", (512, 512)); px, py = (512 - nw) // 2, (512 - nh) // 2
    canvas.paste(img_rs, (px, py)); input_tensor = transforms.ToTensor()(canvas).unsqueeze(0).to(device)
    with torch.no_grad(): outs = [m(input_tensor)[0].cpu().numpy() for m in models]
    ANT_IDX, POST_IDX = [10, 14, 9, 5, 28, 20], [7, 11, 12, 15]
    coords = {}
    for i in range(29):
        hm = outs[1][i] if i in ANT_IDX else (outs[2][i] if i in POST_IDX else outs[0][i])
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        coords[i] = [int((x - px) / ratio), int((y - py) / ratio)]
    gc.collect()
    return coords

# --- ۶. اجرای تحلیل ---
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

if "click_version" not in st.session_state: st.session_state.click_version = 0

if uploaded_file:
    if "raw_img" not in st.session_state or st.session_state.get("file_id") != uploaded_file.name:
        st.session_state.raw_img = Image.open(uploaded_file).convert("RGB")
        st.session_state.file_id = uploaded_file.name
        with st.spinner("🧠 در حال تحلیل..."):
            st.session_state.initial_lms = run_precise_prediction(st.session_state.raw_img, models)
            st.session_state.lms = st.session_state.initial_lms.copy()

    raw_img = st.session_state.raw_img; W, H = raw_img.size
    target_idx = st.sidebar.selectbox("🎯 انتخاب لندمارک:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")
    
    if st.sidebar.button("🔄 Reset Point"):
        st.session_state.lms[target_idx] = st.session_state.initial_lms[target_idx].copy()
        st.session_state.click_version += 1; st.rerun()

    col1, col2 = st.columns([1, 2.5]) # ستون‌ها فشرده‌تر
    with col1:
        st.subheader("🔍 Micro-Adjustment")
        l_pos = st.session_state.lms[target_idx]; size_m = 150 # زوم کمتر
        left, top = max(0, min(int(l_pos[0]-size_m//2), W-size_m)), max(0, min(int(l_pos[1]-size_m//2), H-size_m))
        mag_crop = raw_img.crop((left, top, left+size_m, top+size_m)).resize((300, 300), Image.LANCZOS) # تصویر زوم کوچک‌تر
        mag_draw = ImageDraw.Draw(mag_crop)
        mag_draw.line((135, 150, 165, 150), fill="red", width=2); mag_draw.line((150, 135, 150, 165), fill="red", width=2)
        res_mag = streamlit_image_coordinates(mag_crop, key=f"mag_{target_idx}_{st.session_state.click_version}")
        if res_mag:
            scale_mag = size_m / 300; new_c = [int(left + (res_mag["x"] * scale_mag)), int(top + (res_mag["y"] * scale_mag))]
            if st.session_state.lms[target_idx] != new_c:
                st.session_state.lms[target_idx] = new_c; st.session_state.click_version += 1; st.rerun()

    with col2:
        st.subheader("🖼 نمای گرافیکی")
        # --- اصلاح اندازه تصویر (فشرده‌تر) ---
        disp_w = 700  # عرض کمتر برای تصویر اصلی
        ratio_disp = disp_w / W
        disp_h = int(H * ratio_disp)
        
        draw_img = raw_img.resize((disp_w, disp_h), Image.BILINEAR)
        draw = ImageDraw.Draw(draw_img); l = st.session_state.lms
        def sc(p): return (int(p[0] * ratio_disp), int(p[1] * ratio_disp))
        # -----------------------------

        if all(k in l for k in [10, 4, 0, 2, 18, 22, 17, 21, 15, 5, 14, 3, 20, 21, 23, 17, 8, 27]):
            draw.line([sc(l[10]), sc(l[4])], fill="yellow", width=1) # خطوط نازک‌تر
            draw.line([sc(l[4]), sc(l[0])], fill="cyan", width=1)
            draw.line([sc(l[4]), sc(l[2])], fill="magenta", width=1)
            p_occ_p, p_occ_a = (np.array(l[18]) + np.array(l[22])) / 2, (np.array(l[17]) + np.array(l[21])) / 2
            draw.line([sc(p_occ_p), sc(p_occ_a)], fill="white", width=1)
            draw.line([sc(l[15]), sc(l[5])], fill="orange", width=1)
            draw.line([sc(l[14]), sc(l[3])], fill="purple", width=1)
            draw.line([sc(l[20]), sc(l[21])], fill="blue", width=1)
            draw.line([sc(l[23]), sc(l[17])], fill="green", width=1)
            draw.line([sc(l[8]), sc(l[27])], fill="pink", width=1)

        for i, pos in l.items():
            s_pos = sc(pos); color = (255, 0, 0) if i == target_idx else (0, 255, 0)
            r = 4 if i == target_idx else 2 # نقاط کوچک‌تر
            draw.ellipse([s_pos[0]-r, s_pos[1]-r, s_pos[0]+r, s_pos[1]+r], fill=color, outline="white")
            if text_scale > 1:
                draw.text((s_pos[0]+8, s_pos[1]-4), landmark_names[i], fill=color)

        res_main = streamlit_image_coordinates(draw_img, key=f"main_{st.session_state.click_version}")
        if res_main:
            m_c = [int(res_main["x"] / ratio_disp), int(res_main["y"] / ratio_disp)]
            if st.session_state.lms[target_idx] != m_c:
                st.session_state.lms[target_idx] = m_c; st.session_state.click_version += 1; st.rerun()

    # --- ۷. محاسبات و گزارش (فشرده) ---
    st.divider()
    def get_ang(p1, p2, p3, p4=None):
        v1, v2 = (np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)) if p4 is None else (np.array(p2)-np.array(p1), np.array(p4)-np.array(p3))
        n = np.linalg.norm(v1)*np.linalg.norm(v2); return round(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n if n>0 else 1), -1, 1))), 2)

    def dist_to_line(p, l1, l2):
        v1 = np.append(l2 - l1, 0); v2 = np.append(p - l1, 0)
        return np.linalg.norm(np.cross(v1, v2)) / (np.linalg.norm(l2 - l1) + 1e-6)

    sna, snb = get_ang(l[10], l[4], l[0]), get_ang(l[10], l[4], l[2]); anb = round(sna - snb, 2)
    fma = get_ang(l[15], l[5], l[14], l[3])
    co_a = np.linalg.norm(np.array(l[12])-np.array(l[0])) * pixel_size
    co_gn = np.linalg.norm(np.array(l[12])-np.array(l[13])) * pixel_size
    diff_mcnamara = round(co_gn - co_a, 2)
    p_occ_p, p_occ_a = (np.array(l[18]) + np.array(l[22])) / 2, (np.array(l[17]) + np.array(l[21])) / 2
    v_occ = (p_occ_a - p_occ_p) / (np.linalg.norm(p_occ_a - p_occ_p) + 1e-6)
    wits_mm = (np.dot(np.array(l[0]) - p_occ_p, v_occ) - np.dot(np.array(l[2]) - p_occ_p, v_occ)) * pixel_size
    wits_norm = 0 if gender == "آقا (Male)" else -1
    dist_ls = round(dist_to_line(np.array(l[25]), np.array(l[8]), np.array(l[27])) * pixel_size, 2)
    dist_li = round(dist_to_line(np.array(l[24]), np.array(l[8]), np.array(l[27])) * pixel_size, 2)

    m1, m2, m3, m4 = st.columns(4)
    # استفاده از فونت کوچک‌تر برای اعداد متریک
    m1.metric("Steiner (ANB)", f"{anb}°")
    m2.metric("Wits", f"{round(wits_mm, 2)} mm")
    m3.metric("McNamara", f"{diff_mcnamara} mm")
    m4.metric("Downs (FMA)", f"{fma}°")

    st.divider()
    st.header(f"📑 گزارش بالینی") # عنوان کوتاه‌تر
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("👄 بافت نرم")
        st.write(f"• لب بالا: **{dist_li} mm**")
        st.write(f"• لب پایین: **{dist_ls} mm**")
        st.subheader("💡 درمان")
        w_diff = wits_mm - wits_norm
        diag = "Class II" if w_diff > 1.5 else "Class III" if w_diff < -1.5 else "Class I"
        st.write(f"• وضعیت: **{diag}**")
    with c2:
        st.subheader("📐 زوایا")
        fma_desc = "Vertical" if fma > 32 else "Horizontal" if fma < 20 else "Normal"
        st.write(f"• الگو: **{fma_desc}**")
        st.write(f"• طول فک بالا: {round(co_a, 1)} mm")
        st.write(f"• طول فک پایین: {round(co_gn, 1)} mm")
    
    gc.collect()
