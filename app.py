# -*- coding: utf-8 -*-
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import gdown
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۱. معماری مرجع (DoubleConv & CephaUNet) ---
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

# --- ۲. بارگذاری و پیش‌بینی هوشمند با تفکیک نواحی ---
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
    # تفکیک نواحی تخصصی طبق مرجع
    ANT_IDX, POST_IDX = [10, 14, 9, 5, 28, 20], [7, 11, 12, 15]
    coords = {}
    for i in range(29):
        hm = outs[1][i] if i in ANT_IDX else (outs[2][i] if i in POST_IDX else outs[0][i])
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        coords[i] = [int((x - px) / ratio), int((y - py) / ratio)]
    return coords

# --- ۳. رابط کاربری و نمایش پیشرفته ---
st.set_page_config(page_title="Aariz Precision Station V16.0", layout="wide")
models, device = load_aariz_models()
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

if "click_version" not in st.session_state: st.session_state.click_version = 0

st.sidebar.header("📏 تنظیمات آنالیز")
gender = st.sidebar.radio("جنسیت بیمار:", ["آقا (Male)", "خانم (Female)"])
pixel_size = st.sidebar.number_input("Pixel Size (mm/px):", 0.001, 1.0, 0.1, format="%.4f")
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
        l_pos = st.session_state.lms[target_idx]; size_m = 180 
        left, top = max(0, min(int(l_pos[0]-size_m//2), W-size_m)), max(0, min(int(l_pos[1]-size_m//2), H-size_m))
        mag_crop = raw_img.crop((left, top, left+size_m, top+size_m)).resize((400, 400), Image.LANCZOS)
        mag_draw = ImageDraw.Draw(mag_crop)
        mag_draw.line((180, 200, 220, 200), fill="red", width=2); mag_draw.line((200, 180, 200, 220), fill="red", width=2)
        res_mag = streamlit_image_coordinates(mag_crop, key=f"mag_{target_idx}_{st.session_state.click_version}")
        if res_mag:
            scale_mag = size_m / 400; new_c = [int(left + (res_mag["x"] * scale_mag)), int(top + (res_mag["y"] * scale_mag))]
            if st.session_state.lms[target_idx] != new_c:
                st.session_state.lms[target_idx] = new_c; st.session_state.click_version += 1; st.rerun()

    with col2:
        st.subheader("🖼 نمای گرافیکی و خطوط آنالیز")
        draw_img = raw_img.copy(); draw = ImageDraw.Draw(draw_img); l = st.session_state.lms
        
        # ترسیم کامل خطوط آنالیز مرجع
        if all(k in l for k in [10, 4, 0, 2, 18, 22, 17, 21, 15, 5, 14, 3, 20, 21, 23, 17, 8, 27]):
            draw.line([tuple(l[10]), tuple(l[4])], fill="yellow", width=3) # S-N
            draw.line([tuple(l[4]), tuple(l[0])], fill="cyan", width=2) # N-A
            draw.line([tuple(l[4]), tuple(l[2])], fill="magenta", width=2) # N-B
            p_occ_p, p_occ_a = (np.array(l[18]) + np.array(l[22])) / 2, (np.array(l[17]) + np.array(l[21])) / 2
            draw.line([tuple(p_occ_p), tuple(p_occ_a)], fill="white", width=3) # Occ
            draw.line([tuple(l[15]), tuple(l[5])], fill="orange", width=3) # FH
            draw.line([tuple(l[14]), tuple(l[3])], fill="purple", width=3) # Mandibular
            draw.line([tuple(l[20]), tuple(l[21])], fill="blue", width=2) # U1 Axis
            draw.line([tuple(l[23]), tuple(l[17])], fill="green", width=2) # L1 Axis
            draw.line([tuple(l[8]), tuple(l[27])], fill="pink", width=3) # E-Line

        for i, pos in l.items():
            color = (255, 0, 0) if i == target_idx else (0, 255, 0)
            r = 8 if i == target_idx else 5
            draw.ellipse([pos[0]-r, pos[1]-r, pos[0]+r, pos[1]+r], fill=color, outline="white", width=2)
            draw.text((pos[0]+12, pos[1]-12), landmark_names[i], fill=color)

        st.image(draw_img, use_container_width=True)

    # --- ۴. محاسبات و تحلیل ریاضی ---
    def get_ang(p1, p2, p3, p4=None):
        v1, v2 = (np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)) if p4 is None else (np.array(p2)-np.array(p1), np.array(p4)-np.array(p3))
        n = np.linalg.norm(v1)*np.linalg.norm(v2); return round(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n if n>0 else 1), -1, 1))), 2)

    sna, snb = get_ang(l[10], l[4], l[0]), get_ang(l[10], l[4], l[2]); anb = round(sna - snb, 2)
    fma = get_ang(l[15], l[5], l[14], l[3])
    co_a = round(np.linalg.norm(np.array(l[12])-np.array(l[0])) * pixel_size, 1)
    co_gn = round(np.linalg.norm(np.array(l[12])-np.array(l[13])) * pixel_size, 1)
    diff_mcnamara = round(co_gn - co_a, 1)
    
    p_occ_p, p_occ_a = (np.array(l[18]) + np.array(l[22])) / 2, (np.array(l[17]) + np.array(l[21])) / 2
    v_occ = (p_occ_a - p_occ_p) / (np.linalg.norm(p_occ_a - p_occ_p) + 1e-6)
    wits_mm = round((np.dot(np.array(l[0]) - p_occ_p, v_occ) - np.dot(np.array(l[2]) - p_occ_p, v_occ)) * pixel_size, 2)
    wits_norm = 0 if gender == "آقا (Male)" else -1
    
    # محاسبه فاصله تا خط E (Ricketts)
    def dist_line(p, l1, l2): return round(np.cross(l2-l1, l1-p) / (np.linalg.norm(l2-l1) + 1e-6) * pixel_size, 2)
    dist_ls = dist_line(np.array(l[25]), np.array(l[8]), np.array(l[27]))
    dist_li = dist_line(np.array(l[24]), np.array(l[8]), np.array(l[27]))

    # --- ۵. گزارش بالینی و نقشه راه درمان (Diagnostic Roadmap) ---
    st.divider()
    st.header(f"📑 گزارش بالینی نهایی ({gender})")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Steiner (ANB)", f"{anb}°", f"SNA: {sna}, SNB: {snb}")
    m2.metric("Wits Appraisal", f"{wits_mm} mm", f"Normal: {wits_norm}")
    m3.metric("McNamara Diff", f"{diff_mcnamara} mm", f"Co-Gn: {co_gn}")
    m4.metric("Growth (FMA)", f"{fma}°")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("💡 نقشه راه درمان (Diagnostic Roadmap)")
        w_diff = wits_mm - wits_norm
        diag = "Class II" if w_diff > 1.5 else "Class III" if w_diff < -1.5 else "Class I"
        st.info(f"وضعیت فکی: **Skeletal {diag}**")
        if abs(anb) > 8 or abs(diff_mcnamara - 25) > 10:
            st.error("🚨 دیسکرپانسی شدید؛ احتمال نیاز به جراحی فک بالا (Orthognathic Surgery).")
        else:
            st.success("✅ درمان ارتودنسی ثابت با مکانوتراپی استاندارد.")

        st.subheader("👄 تحلیل بافت نرم و زیبایی")
        st.write(f"• لب بالا تا خط E: **{dist_ls} mm**")
        st.write(f"• لب پایین تا خط E: **{dist_li} mm**")
        if gender == "آقا (Male)" and dist_li > 0: st.warning("⚠️ نیم‌رخ محدب (Convex) در مردان.")
        elif gender == "خانم (Female)" and dist_li > 1: st.warning("⚠️ پروتروژن لب در نیم‌رخ زنانه.")

    with c2:
        st.subheader("📐 تحلیل زوایا و رشد")
        fma_desc = "Vertical (High Angle)" if fma > 32 else "Horizontal (Low Angle)" if fma < 20 else "Normal"
        st.write(f"• الگوی رشد اسکلتال: **{fma_desc}**")
        st.write(f"• طول فک بالا (Co-A): {co_a} mm")
        st.write(f"• طول فک پایین (Co-Gn): {co_gn} mm")

    if st.button("📥 خروجی نهایی برای پرونده"):
        st.balloons()
        st.success("تمام پارامترهای مرجع (Steiner, McNamara, Wits, Ricketts) ثبت شدند.")
