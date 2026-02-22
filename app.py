import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import gdown
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates
import base64

# بررسی وجود fpdf برای گزارش‌گیری (افزایشی)
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# --- ۱. معماری مرجع Aariz (بدون هیچ تغییری) ---
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

# --- ۲. بارگذاری مدل‌ها (تکرار دقیق مرجع) ---
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
            m = CephaUNet(n_landmarks=29).to(device); ckpt = torch.load(f, map_location=device)
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
    ANT_IDX, POST_IDX = [10, 14, 9, 5, 28, 20], [7, 11, 12, 15]
    coords = {}
    for i in range(29):
        hm = outs[1][i] if i in ANT_IDX else (outs[2][i] if i in POST_IDX else outs[0][i])
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        coords[i] = [int((x - px) / ratio), int((y - py) / ratio)]
    return coords

# --- ۳. رابط کاربری (UI) و تنظیمات ---
st.set_page_config(page_title="Aariz Precision Station V7.0", layout="wide")
models, device = load_aariz_models()
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

if "click_version" not in st.session_state: st.session_state.click_version = 0

st.sidebar.header("📏 Clinical Settings")
patient_name = st.sidebar.text_input("Patient Name:", "Aariz Case")
gender = st.sidebar.radio("Gender:", ["Male", "Female"])
pixel_size = st.sidebar.number_input("Pixel Size (mm/px):", 0.01, 1.0, 0.1, format="%.4f")
text_scale = st.sidebar.slider("Text Scale:", 1, 10, 3)

uploaded_file = st.sidebar.file_uploader("Upload Cephalogram:", type=['png', 'jpg', 'jpeg'])

if uploaded_file and len(models) == 3:
    raw_img = Image.open(uploaded_file).convert("RGB"); W, H = raw_img.size
    if "lms" not in st.session_state or st.session_state.get("file_id") != uploaded_file.name:
        st.session_state.initial_lms = run_precise_prediction(raw_img, models, device)
        st.session_state.lms = st.session_state.initial_lms.copy(); st.session_state.file_id = uploaded_file.name

    target_idx = st.sidebar.selectbox("🎯 Active Landmark:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")
    
    col1, col2 = st.columns([1.2, 2.5])
    with col1:
        st.subheader("🔍 Anti-Drift Magnifier")
        l_pos = st.session_state.lms[target_idx]; size_m = 180 
        left, top = max(0, min(int(l_pos[0]-size_m//2), W-size_m)), max(0, min(int(l_pos[1]-size_m//2), H-size_m))
        mag_crop = raw_img.crop((left, top, left+size_m, top+size_m)).resize((400, 400), Image.LANCZOS)
        mag_draw = ImageDraw.Draw(mag_crop)
        mag_draw.line((180, 200, 220, 200), fill="red", width=3); mag_draw.line((200, 180, 200, 220), fill="red", width=3)
        res_mag = streamlit_image_coordinates(mag_crop, key=f"mag_{target_idx}_{st.session_state.click_version}")
        if res_mag:
            scale_mag = size_m / 400; new_c = [int(left + (res_mag["x"] * scale_mag)), int(top + (res_mag["y"] * scale_mag))]
            if st.session_state.lms[target_idx] != new_c:
                st.session_state.lms[target_idx] = new_c; st.session_state.click_version += 1; st.rerun()

    with col2:
        st.subheader("🖼 Full Cephalometric Overlay")
        draw_img = raw_img.copy(); draw = ImageDraw.Draw(draw_img); l = st.session_state.lms
        
        # --- بخش ترسیم گرافیکی جامع (Steiner + McNamara + Teeth) ---
        if all(k in l for k in [10, 4, 0, 2, 18, 22, 17, 21, 15, 5, 14, 3, 20, 21, 23, 17, 8, 27, 12, 13]):
            # Steiner & Occ Plane
            draw.line([tuple(l[10]), tuple(l[4])], fill="yellow", width=3) # SN
            draw.line([tuple(l[4]), tuple(l[0])], fill="cyan", width=2)   # NA
            draw.line([tuple(l[4]), tuple(l[2])], fill="magenta", width=2) # NB
            p_occ_p, p_occ_a = (np.array(l[18]) + np.array(l[22])) / 2, (np.array(l[17]) + np.array(l[21])) / 2
            draw.line([tuple(p_occ_p), tuple(p_occ_a)], fill="white", width=3) # Occ Plane
            draw.line([tuple(l[20]), tuple(l[21])], fill="blue", width=2)  # U1 Axis
            draw.line([tuple(l[23]), tuple(l[17])], fill="green", width=2) # L1 Axis
            # FH & McNamara
            draw.line([tuple(l[15]), tuple(l[5])], fill="orange", width=3) # FH
            draw.line([tuple(l[14]), tuple(l[3])], fill="purple", width=3) # MP
            draw.line([tuple(l[12]), tuple(l[0])], fill="brown", width=2)  # Co-A
            draw.line([tuple(l[12]), tuple(l[13])], fill="brown", width=2) # Co-Gn
            draw.line([tuple(l[8]), tuple(l[27])], fill="pink", width=3)   # E-Line

        for i, pos in l.items():
            color = (255, 0, 0) if i == target_idx else (0, 255, 0)
            draw.ellipse([pos[0]-7, pos[1]-7, pos[0]+7, pos[1]+7], fill=color, outline="white", width=2)

        streamlit_image_coordinates(draw_img, width=850, key=f"main_{st.session_state.click_version}")

    # --- ۴. محاسبات تجمعی (تلفیق Steiner, Wits, McNamara) ---
    st.divider()
    def get_ang(p1, p2, p3, p4=None):
        v1, v2 = (np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)) if p4 is None else (np.array(p2)-np.array(p1), np.array(p4)-np.array(p3))
        n = np.linalg.norm(v1)*np.linalg.norm(v2); return round(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n if n>0 else 1), -1, 1))), 1)
    
    sna, snb = get_ang(l[10], l[4], l[0]), get_ang(l[10], l[4], l[2]); anb = round(sna - snb, 1)
    fma = get_ang(l[15], l[5], l[14], l[3])
    co_a, co_gn = np.linalg.norm(np.array(l[12])-np.array(l[0]))*pixel_size, np.linalg.norm(np.array(l[12])-np.array(l[13]))*pixel_size
    diff = round(co_gn - co_a, 1)
    
    # Wits Calibration
    p_occ_p, p_occ_a = (np.array(l[18]) + np.array(l[22])) / 2, (np.array(l[17]) + np.array(l[21])) / 2
    v_occ = (p_occ_a - p_occ_p) / (np.linalg.norm(p_occ_a - p_occ_p) + 1e-6)
    wits = (np.dot(np.array(l[0]) - p_occ_p, v_occ) - np.dot(np.array(l[2]) - p_occ_p, v_occ)) * pixel_size

    # نمایش مقادیر در متریک‌ها
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("ANB (Steiner)", f"{anb}°")
    col_m2.metric("Wits (Occ)", f"{round(wits, 1)} mm")
    col_m3.metric("McNamara Diff", f"{diff} mm")
    col_m4.metric("Growth (FMA)", f"{fma}°")

    # --- ۵. گزارش بالینی و طرح درمان (بخش حیاتی بازگشتی) ---
    st.divider()
    st.header("📑 Clinical Analysis & Treatment Roadmap")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("💡 Skeletal Diagnosis")
        wits_norm = 0 if gender == "Male" else -1
        class_status = "II" if wits > wits_norm + 2 else "III" if wits < wits_norm - 2 else "I"
        st.write(f"• **Skeletal Relationship:** Class {class_status}")
        if abs(anb) > 7 or abs(diff - 25) > 12:
            st.error("🚨 **High Severity:** Consider Orthognathic Surgery assessment.")
        else:
            st.success("✅ **Moderate Severity:** Camouflage or Growth Modification possible.")

    with c2:
        st.subheader("👄 Aesthetic & Soft Tissue")
        dist_ls = round(np.cross(np.array(l[27])-np.array(l[8]), np.array(l[8])-np.array(l[25])) / (np.linalg.norm(np.array(l[27])-np.array(l[8])) + 1e-6) * pixel_size, 1)
        dist_li = round(np.cross(np.array(l[27])-np.array(l[8]), np.array(l[8])-np.array(l[24])) / (np.linalg.norm(np.array(l[27])-np.array(l[8])) + 1e-6) * pixel_size, 1)
        st.write(f"• Upper Lip to E-Line: **{dist_ls} mm**")
        st.write(f"• Lower Lip to E-Line: **{dist_li} mm**")

    # --- ۶. خروجی PDF (بخش امن افزایشی) ---
    st.sidebar.divider()
    if not FPDF_AVAILABLE:
        st.sidebar.warning("⚠️ Install 'fpdf' via requirements.txt for PDF reports.")
    elif st.sidebar.button("📥 Generate Full PDF Report"):
        pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Aariz Precision Station - Final Report", ln=True, align='C')
        pdf.ln(10); pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Patient: {patient_name} | Gender: {gender}", ln=True)
        pdf.cell(200, 10, txt=f"Diagnosis: Skeletal Class {class_status}", ln=True)
        pdf.cell(200, 10, txt=f"ANB: {anb} | Wits: {round(wits,1)} | McNamara Diff: {diff}mm", ln=True)
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        b64 = base64.b64encode(pdf_bytes).decode()
        st.sidebar.markdown(f'<a href="data:application/pdf;base64,{b64}" download="Report_{patient_name}.pdf">Download PDF</a>', unsafe_allow_html=True)
