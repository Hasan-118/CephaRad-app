import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import gdown
import gc
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates
from fpdf import FPDF
import base64

# --- ۱. معماری مرجع Aariz (V7.8.16 - بدون هیچ تغییری) ---
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

# --- ۲. لودر و توابع پیش‌بینی (V7.8.16) ---
@st.cache_resource
def load_aariz_models():
    model_ids = {'checkpoint_unet_clinical.pth': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks', 'specialist_pure_model.pth': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU', 'tmj_specialist_model.pth': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'}
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
    coords = {i: [int((np.unravel_index(np.argmax(outs[1][i] if i in ANT_IDX else (outs[2][i] if i in POST_IDX else outs[0][i])), (512,512))[1] - px) / ratio), int((np.unravel_index(np.argmax(outs[1][i] if i in ANT_IDX else (outs[2][i] if i in POST_IDX else outs[0][i])), (512,512))[0] - py) / ratio)] for i in range(29)}
    gc.collect(); return coords

# --- ۳. رابط کاربری (V7.8.16) ---
st.set_page_config(page_title="Aariz Precision Station V7.8.20", layout="wide")
models, device = load_aariz_models()
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

if "click_version" not in st.session_state: st.session_state.click_version = 0

st.sidebar.header("📏 تنظیمات بیمار")
p_name = st.sidebar.text_input("نام بیمار:", "Patient_001")
gender = st.sidebar.radio("جنسیت بیمار:", ["آقا (Male)", "خانم (Female)"])
pixel_size = st.sidebar.number_input("Pixel Size (mm/px):", 0.01, 1.0, 0.1, 0.001, format="%.4f")
text_scale = st.sidebar.slider("🔤 مقیاس نام لندمارک:", 1, 10, 3)

uploaded_file = st.sidebar.file_uploader("آپلود تصویر سفالومتری:", type=['png', 'jpg', 'jpeg'])

if uploaded_file and len(models) == 3:
    raw_img = Image.open(uploaded_file).convert("RGB"); W, H = raw_img.size
    if "lms" not in st.session_state or st.session_state.get("file_id") != uploaded_file.name:
        st.session_state.initial_lms = run_precise_prediction(raw_img, models, device)
        st.session_state.lms = st.session_state.initial_lms.copy(); st.session_state.file_id = uploaded_file.name

    target_idx = st.sidebar.selectbox("🎯 انتخاب لندمارک فعال:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")
    
    col1, col2 = st.columns([1.2, 2.5])
    with col1:
        st.subheader("🔍 Micro-Adjustment")
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
        st.subheader("🖼 نمای گرافیکی و خطوط آنالیز")
        draw_img = raw_img.copy(); draw = ImageDraw.Draw(draw_img); l = st.session_state.lms
        if all(k in l for k in [10, 4, 0, 2, 15, 5, 14, 3, 8, 27, 12, 13]):
            draw.line([tuple(l[10]), tuple(l[4])], fill="yellow", width=3) # S-N
            draw.line([tuple(l[4]), tuple(l[0])], fill="cyan", width=2) # N-A
            draw.line([tuple(l[4]), tuple(l[2])], fill="magenta", width=2) # N-B
            draw.line([tuple(l[15]), tuple(l[5])], fill="orange", width=3) # FH
            draw.line([tuple(l[14]), tuple(l[3])], fill="purple", width=3) # Mandibular
            draw.line([tuple(l[8]), tuple(l[27])], fill="pink", width=3) # E-Line

        for i, pos in l.items():
            color = (255, 0, 0) if i == target_idx else (0, 255, 0)
            r = 10 if i == target_idx else 6
            draw.ellipse([pos[0]-r, pos[1]-r, pos[0]+r, pos[1]+r], fill=color, outline="white", width=2)
            name_text = landmark_names[i]
            temp_txt = Image.new('RGBA', (len(name_text)*8, 12), (0,0,0,0))
            ImageDraw.Draw(temp_txt).text((0, 0), name_text, fill=color)
            scaled_txt = temp_txt.resize((int(temp_txt.width*text_scale), int(temp_txt.height*text_scale)), Image.NEAREST)
            draw_img.paste(scaled_txt, (pos[0]+r+10, pos[1]-r), scaled_txt)

        res_main = streamlit_image_coordinates(draw_img, width=850, key=f"main_{st.session_state.click_version}")
        if res_main:
            c_scale = W / 850; m_c = [int(res_main["x"] * c_scale), int(res_main["y"] * c_scale)]
            if st.session_state.lms[target_idx] != m_c:
                st.session_state.lms[target_idx] = m_c; st.session_state.click_version += 1; st.rerun()

    # --- ۴. محاسبات (V7.8.16 - بدون تغییر یک عدد) ---
    st.divider()
    def get_ang(p1, p2, p3, p4=None):
        v1, v2 = (np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)) if p4 is None else (np.array(p2)-np.array(p1), np.array(p4)-np.array(p3))
        n = np.linalg.norm(v1)*np.linalg.norm(v2); return round(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n if n>0 else 1), -1, 1))), 2)
    
    def dist_to_line(p, l1, l2):
        p3d, l1_3d, l2_3d = np.append(p, 0), np.append(l1, 0), np.append(l2, 0)
        return np.linalg.norm(np.cross(l2_3d-l1_3d, l1_3d-p3d)) / (np.linalg.norm(l2_3d-l1_3d) + 1e-6)

    sna, snb = get_ang(l[10], l[4], l[0]), get_ang(l[10], l[4], l[2])
    anb = round(sna - snb, 2); fma = get_ang(l[15], l[5], l[14], l[3])
    co_a = round(np.linalg.norm(np.array(l[12])-np.array(l[0])) * pixel_size, 2)
    co_gn = round(np.linalg.norm(np.array(l[12])-np.array(l[13])) * pixel_size, 2)
    diff_mcnamara = round(co_gn - co_a, 2)
    dist_ls = round(dist_to_line(np.array(l[25]), np.array(l[8]), np.array(l[27])) * pixel_size, 2)
    dist_li = round(dist_to_line(np.array(l[24]), np.array(l[8]), np.array(l[27])) * pixel_size, 2)
    uit_angle, lit_angle = get_ang(l[10], l[4], l[21], l[20]), get_ang(l[14], l[3], l[23], l[17])
    
    diag = "Class II" if anb > 4 else "Class III" if anb < 0 else "Class I"
    fma_desc = "Vertical Growth" if fma > 30 else "Horizontal Growth" if fma < 20 else "Normal Growth"

    # --- ۵. گزارش دهی و PDF (Safe Version) ---
    def create_pdf_report():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", 'B', 16)
        pdf.cell(0, 15, "Aariz Precision Station Report", ln=True, align='C')
        pdf.set_font("Helvetica", '', 10)
        
        # پاکسازی متغیرها از کاراکترهای یونیکد برای جلوگیری از خطای latin-1
        safe_name = p_name.encode('ascii', 'ignore').decode('ascii')
        safe_gender = gender.encode('ascii', 'ignore').decode('ascii')
        
        pdf.cell(0, 10, f"Patient: {safe_name} | Gender: {safe_gender}", ln=True, align='C')
        pdf.ln(5)
        
        # ترسیم جدول
        pdf.set_font("Helvetica", 'B', 10)
        pdf.set_fill_color(230, 230, 230)
        cols = [45, 35, 45, 65]
        h = ["Parameter", "Value", "Norm", "Interpretation"]
        for i, text in enumerate(h): pdf.cell(cols[i], 10, text, 1, 0, 'C', True)
        pdf.ln()
        
        pdf.set_font("Helvetica", '', 10)
        rows = [
            ["SNA Angle", f"{sna} deg", "82 +/- 2", "Maxilla Pos"],
            ["SNB Angle", f"{snb} deg", "80 +/- 2", "Mandible Pos"],
            ["ANB Angle", f"{anb} deg", "2.0 +/- 2", diag],
            ["FMA Angle", f"{fma} deg", "25 +/- 5", fma_desc],
            ["McNamara Diff", f"{diff_mcnamara} mm", "25-30 mm", "Balance"],
            ["U1 to SN", f"{uit_angle} deg", "102 +/- 2", "U-Inclination"],
            ["IMPA (L1-MP)", f"{lit_angle} deg", "90 +/- 3", "L-Inclination"],
            ["Upper Lip to E", f"{dist_ls} mm", "-2 to -4 mm", "Esthetic"],
            ["Lower Lip to E", f"{dist_li} mm", "0 to -2 mm", "Esthetic"]
        ]
        for row in rows:
            for i, val in enumerate(row): pdf.cell(cols[i], 10, str(val), 1)
            pdf.ln()

        pdf.ln(5)
        pdf.set_font("Helvetica", 'B', 11); pdf.cell(0, 10, "Clinical Roadmap:", ln=True)
        pdf.set_font("Helvetica", '', 10)
        roadmap = "Severe discrepancy. Surgery potential." if abs(anb) > 8 else "Moderate discrepancy. Camouflage possible."
        pdf.multi_cell(0, 8, roadmap)
        
        # بخش نهایی خروجی با مدیریت خطا
        try:
            return pdf.output(dest='S').encode('latin-1')
        except UnicodeEncodeError:
            # اگر باز هم خطا داد، کل متن را به ascii تقلیل بده
            return pdf.output(dest='S').encode('ascii', 'replace')

    st.header(f"📑 گزارش و تفسیر بالینی ({gender})")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🦷 تحلیل اسکلتال و دندانی")
        st.metric("ANB Angle", f"{anb} deg", f"SNA: {sna} / SNB: {snb}")
        st.metric("McNamara Difference", f"{diff_mcnamara} mm", "Normal: 25-30mm")
        st.metric("U1 to SN / IMPA", f"{uit_angle} deg / {lit_angle} deg")
        st.info(f"**تشخیص اسکلتال:** {diag}")

    with c2:
        st.subheader("👄 زیبایی و بافت نرم")
        st.write(f"• فاصله لب بالا تا خط E: **{dist_ls} mm**")
        st.write(f"• فاصله لب پایین تا خط E: **{dist_li} mm**")
        st.warning(f"**الگوی رشد:** {fma_desc} ({fma} deg)")
        
        if st.download_button("📥 دانلود گزارش PDF حرفه‌ای", create_pdf_report(), f"Report_{p_name}.pdf", "application/pdf"):
            st.success("گزارش با موفقیت تولید شد.")
