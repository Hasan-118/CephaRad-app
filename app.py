import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import gc
import io
import urllib.request
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

# وارد کردن کتابخانه‌های ReportLab برای ساخت PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# کتابخانه‌های پردازش متن فارسی برای ReportLab
import arabic_reshaper
from bidi.algorithm import get_display

# وارد کردن ماژول اسکن داخل دهانی ۳D
try:
    from intraoral_3d_module import render_intraoral_3d_tab
except ImportError:
    render_intraoral_3d_tab = None

# --- آماده‌سازی و ثبت فونت Vazir برای ReportLab ---
def register_vazir_font():
    font_path = "Vazir.ttf"
    if not os.path.exists(font_path):
        url = "https://github.com/rastikerdar/vazir-font/raw/master/dist/Vazir.ttf"
        try:
            urllib.request.urlretrieve(url, font_path)
        except Exception:
            pass
    if os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont('Vazir', font_path))
            return 'Vazir'
        except Exception:
            return 'Helvetica'
    return 'Helvetica'

VAZIR_FONT_NAME = register_vazir_font()

def reshape_fa(text):
    if VAZIR_FONT_NAME == 'Helvetica' or not text:
        return text
    reshaped_text = arabic_reshaper.reshape(str(text))
    return get_display(reshaped_text)

# --- ۱. تنظیمات صفحه و استایل ---
st.set_page_config(page_title="Aariz Precision Station V7.8.16", layout="wide")

st.markdown("""
<style>
    html, body, [class*="css"]  { font-size: 14px; }
    .stButton>button { padding: 0.2rem 0.5rem; font-size: 12px; }
    .stSelectbox, .stRadio, .stNumberInput, .stFileUploader { margin-top: -10px; }
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

# --- ۳. مدیریت فایل و بارگذاری مدل‌ها ---
def get_model_map():
    return ['checkpoint_unet_clinical_int8.pth', 'specialist_pure_model_int8.pth', 'tmj_specialist_model_int8.pth']

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
        m = torch.quantization.quantize_dynamic(
            m, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
        ckpt = torch.load(f, map_location=device)
        m.load_state_dict(ckpt)
        m.eval()
        loaded_models.append(m)
        del ckpt
        gc.collect()
    return loaded_models

# --- ماژول ساخت PDF افزایشی چندصفحه‌ای با پشتیبانی فونت Vazir ---
def generate_clinical_pdf(patient_info, norm_table_data, detailed_interpretations, treatment_plan, annotated_img_bytes):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontName=VAZIR_FONT_NAME, fontSize=16, leading=20,
        textColor=colors.HexColor('#1E3A8A'), alignment=1, spaceAfter=8
    )
    section_style = ParagraphStyle(
        'SectionHeader', parent=styles['Heading2'], fontName=VAZIR_FONT_NAME, fontSize=12, leading=15,
        textColor=colors.HexColor('#1E40AF'), spaceBefore=8, spaceAfter=4, alignment=2
    )
    normal_style = ParagraphStyle(
        'DocNormal', parent=styles['Normal'], fontName=VAZIR_FONT_NAME, fontSize=10, leading=14, alignment=2
    )
    
    elements = []
    
    elements.append(Paragraph(reshape_fa("Aariz Precision Station - Comprehensive Cephalometric Report"), title_style))
    elements.append(Spacer(1, 6))
    
    info_data = [
        [Paragraph(reshape_fa(f"<b>جنسیت بیمار:</b> {patient_info.get('gender')}"), normal_style),
         Paragraph(f"<b>Pixel Size:</b> {patient_info.get('pixel_size')} mm/px", normal_style)],
        [Paragraph(f"<b>Date:</b> {patient_info.get('date')}", normal_style),
         Paragraph(reshape_fa(f"<b>تشخیص اولیه:</b> {patient_info.get('diag')}"), normal_style)]
    ]
    info_table = Table(info_data, colWidths=[270, 270])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F3F4F6')),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LINEBELOW', (0, -1), (-1, -1), 1, colors.HexColor('#E5E7EB')),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 8))
    
    elements.append(Paragraph(reshape_fa("۱. مقایسه با پارامترهای مرجع و استاندارد (Norms)"), section_style))
    table_content = [[
        Paragraph(reshape_fa("وضعیت بالینی"), normal_style),
        Paragraph(reshape_fa("انحراف"), normal_style),
        Paragraph(reshape_fa("نرمال / میانگین"), normal_style),
        Paragraph(reshape_fa("مقدار اندازه‌گیری"), normal_style),
        Paragraph(reshape_fa("پارامتر"), normal_style)
    ]]
    for item in norm_table_data:
        table_content.append([
            Paragraph(reshape_fa(item['status']), normal_style),
            Paragraph(f"{item['dev']} {item['unit']}", normal_style),
            Paragraph(f"{item['norm']} {item['unit']}", normal_style),
            Paragraph(f"{item['measured']} {item['unit']}", normal_style),
            Paragraph(item['param'], normal_style)
        ])
        
    angles_table = Table(table_content, colWidths=[130, 90, 100, 90, 130])
    angles_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E40AF')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), VAZIR_FONT_NAME),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')]),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    elements.append(angles_table)
    elements.append(Spacer(1, 8))
    
    elements.append(Paragraph(reshape_fa("۲. تفسیر تفصیلی و تشخیصی"), section_style))
    interp_text = ""
    for category, desc in detailed_interpretations.items():
        interp_text += f"<b>• {reshape_fa(category)}:</b> {reshape_fa(desc)}<br/><br/>"
    elements.append(Paragraph(interp_text, normal_style))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(reshape_fa("۳. طرح درمان پیشنهادی بالینی"), section_style))
    plan_text = ""
    for idx, item in enumerate(treatment_plan, 1):
        plan_text += f"<b>{idx}. {reshape_fa(item['title'])}:</b> {reshape_fa(item['desc'])}<br/>"
    elements.append(Paragraph(plan_text, normal_style))
    elements.append(Spacer(1, 8))
    
    if annotated_img_bytes:
        elements.append(Paragraph(reshape_fa("۴. تصویر سفلومتری و لندمارک‌های رسم‌شده"), section_style))
        img_buf = io.BytesIO(annotated_img_bytes)
        rl_img = RLImage(img_buf, width=220, height=220)
        elements.append(rl_img)
        
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# --- ۴. تفکیک رابط کاربری به تب‌های اصلی ---
tab_ceph, tab_3d = st.tabs(["📐 آنالیز سئفالومتری (۲D)", "🦷 آنالیز اسکن داخل دهانی (۳D)"])

with tab_ceph:
    # --- اجرای منطق اصلی برنامه سئفالومتری ---
    models = load_models()

    st.sidebar.title("🛠 مرکز پردازش Aariz")
    gender = st.sidebar.radio("جنسیت بیمار:", ["آقا (Male)", "خانم (Female)"])
    pixel_size = st.sidebar.number_input("Pixel Size (mm/px):", 0.01, 1.0, 0.1, 0.001, format="%.4f")
    text_scale = st.sidebar.slider("🔤 مقیاس نام لندمارک:", 1, 10, 2)
    uploaded_file = st.sidebar.file_uploader("آپلود تصویر:", type=['png', 'jpg', 'jpeg'])

    if models is None:
        st.stop()

    # --- ۵. پردازش تصویر ---
    def run_precise_prediction(img_pil, models):
        device = torch.device("cpu")
        ow, oh = img_pil.size; img_gray = img_pil.convert('L'); ratio = 512 / max(ow, oh)
        nw, nh = int(ow * ratio), int(oh * ratio); img_rs = img_gray.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("L", (512, 512)); px, py = (512 - nw) // 2, (512 - nh) // 2
        canvas.paste(img_rs, (px, py))
        
        np_img = np.array(canvas).astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(np_img).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outs = []
            for m in models:
                out = m(input_tensor)[0].cpu().numpy()
                outs.append(out)
                del out
                gc.collect()
                
            ANT_IDX, POST_IDX = [10, 14, 9, 5, 28, 20], [7, 11, 12, 15]
            coords = {}
            for i in range(29):
                hm = outs[1][i] if i in ANT_IDX else (outs[2][i] if i in POST_IDX else outs[0][i])
                y, x = np.unravel_index(np.argmax(hm), hm.shape)
                coords[i] = [int((x - px) / ratio), int((y - py) / ratio)]
        
        del input_tensor
        del outs
        gc.collect()
        return coords

    # --- ۶. نمایش و تنظیم لندمارک‌ها ---
    landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

    if "click_version" not in st.session_state: st.session_state.click_version = 0

    if uploaded_file:
        if "raw_img" not in st.session_state or st.session_state.get("file_id") != uploaded_file.name:
            st.session_state.raw_img = Image.open(uploaded_file).convert("RGB")
            st.session_state.file_id = uploaded_file.name
            with st.spinner("🧠 در حال تحلیل با مدل‌های بهینه‌شده..."):
                st.session_state.initial_lms = run_precise_prediction(st.session_state.raw_img, models)
                st.session_state.lms = st.session_state.initial_lms.copy()

        raw_img = st.session_state.raw_img; W, H = raw_img.size
        target_idx = st.sidebar.selectbox("🎯 انتخاب لندمارک:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")
        
        if st.sidebar.button("🔄 Reset Point"):
            st.session_state.lms[target_idx] = st.session_state.initial_lms[target_idx].copy()
            st.session_state.click_version += 1; st.rerun()

        col1, col2 = st.columns([1, 2.5])
        with col1:
            st.subheader("🔍 Micro-Adjustment")
            l_pos = st.session_state.lms[target_idx]; size_m = 150
            left, top = max(0, min(int(l_pos[0]-size_m//2), W-size_m)), max(0, min(int(l_pos[1]-size_m//2), H-size_m))
            mag_crop = raw_img.crop((left, top, left+size_m, top+size_m)).resize((300, 300), Image.LANCZOS)
            mag_draw = ImageDraw.Draw(mag_crop)
            mag_draw.line((135, 150, 165, 150), fill="red", width=2); mag_draw.line((150, 135, 150, 165), fill="red", width=2)
            res_mag = streamlit_image_coordinates(mag_crop, key=f"mag_{target_idx}_{st.session_state.click_version}")
            if res_mag:
                scale_mag = size_m / 300; new_c = [int(left + (res_mag["x"] * scale_mag)), int(top + (res_mag["y"] * scale_mag))]
                if st.session_state.lms[target_idx] != new_c:
                    st.session_state.lms[target_idx] = new_c; st.session_state.click_version += 1; st.rerun()

        with col2:
            st.subheader("🖼 نمای گرافیکی")
            disp_w = 700
            ratio_disp = disp_w / W
            disp_h = int(H * ratio_disp)
            
            draw_img = raw_img.resize((disp_w, disp_h), Image.BILINEAR)
            draw = ImageDraw.Draw(draw_img); l = st.session_state.lms
            def sc(p): return (int(p[0] * ratio_disp), int(p[1] * ratio_disp))

            if all(k in l for k in [10, 4, 0, 2, 18, 22, 17, 21, 15, 5, 14, 3, 20, 21, 23, 17, 8, 27]):
                draw.line([sc(l[10]), sc(l[4])], fill="yellow", width=1)
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
                r = 4 if i == target_idx else 2
                draw.ellipse([s_pos[0]-r, s_pos[1]-r, s_pos[0]+r, s_pos[1]+r], fill=color, outline="white")
                if text_scale > 1:
                    draw.text((s_pos[0]+8, s_pos[1]-4), landmark_names[i], fill=color)

            res_main = streamlit_image_coordinates(draw_img, key=f"main_{st.session_state.click_version}")
            if res_main:
                m_c = [int(res_main["x"] / ratio_disp), int(res_main["y"] / ratio_disp)]
                if st.session_state.lms[target_idx] != m_c:
                    st.session_state.lms[target_idx] = m_c; st.session_state.click_version += 1; st.rerun()

        # --- ۷. محاسبات کامل زاویه‌ای و طولی ---
        st.divider()
        def get_ang(p1, p2, p3, p4=None):
            v1, v2 = (np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)) if p4 is None else (np.array(p2)-np.array(p1), np.array(p4)-np.array(p3))
            n = np.linalg.norm(v1)*np.linalg.norm(v2); return round(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n if n>0 else 1), -1, 1))), 2)

        def dist_to_line(p, l1, l2):
            v1 = np.append(l2 - l1, 0); v2 = np.append(p - l1, 0)
            return np.linalg.norm(np.cross(v1, v2)) / (np.linalg.norm(l2 - l1) + 1e-6)

        sna = get_ang(l[10], l[4], l[0])
        snb = get_ang(l[10], l[4], l[2])
        anb = round(sna - snb, 2)
        fma = get_ang(l[15], l[5], l[14], l[3])
        co_a = round(np.linalg.norm(np.array(l[12])-np.array(l[0])) * pixel_size, 1)
        co_gn = round(np.linalg.norm(np.array(l[12])-np.array(l[13])) * pixel_size, 1)
        diff_mcnamara = round(co_gn - co_a, 2)
        
        p_occ_p, p_occ_a = (np.array(l[18]) + np.array(l[22])) / 2, (np.array(l[17]) + np.array(l[21])) / 2
        v_occ = (p_occ_a - p_occ_p) / (np.linalg.norm(p_occ_a - p_occ_p) + 1e-6)
        wits_mm = round((np.dot(np.array(l[0]) - p_occ_p, v_occ) - np.dot(np.array(l[2]) - p_occ_p, v_occ)) * pixel_size, 2)
        wits_norm = 0.0 if gender == "آقا (Male)" else -1.0
        
        dist_ls = round(dist_to_line(np.array(l[25]), np.array(l[8]), np.array(l[27])) * pixel_size, 2)
        dist_li = round(dist_to_line(np.array(l[24]), np.array(l[8]), np.array(l[27])) * pixel_size, 2)

        # --- ۸. ساخت جدول مقایسه با NORMها و انحرافات ---
        norm_table_data = [
            {"param": "SNA (Maxilla Pos)", "measured": sna, "unit": "°", "norm": 82.0, "dev": round(sna - 82.0, 2), "status": "Protrusive" if sna > 84 else ("Retrusive" if sna < 80 else "Normal")},
            {"param": "SNB (Mandible Pos)", "measured": snb, "unit": "°", "norm": 80.0, "dev": round(snb - 80.0, 2), "status": "Protrusive" if snb > 82 else ("Retrusive" if snb < 78 else "Normal")},
            {"param": "ANB (Skeletal Rel)", "measured": anb, "unit": "°", "norm": 2.0, "dev": round(anb - 2.0, 2), "status": "Class II" if anb > 4 else ("Class III" if anb < 0 else "Class I")},
            {"param": "Wits Appraisal", "measured": wits_mm, "unit": "mm", "norm": wits_norm, "dev": round(wits_mm - wits_norm, 2), "status": "Class II" if (wits_mm - wits_norm) > 1.5 else ("Class III" if (wits_mm - wits_norm) < -1.5 else "Class I")},
            {"param": "FMA (Vertical Angle)", "measured": fma, "unit": "°", "norm": 25.0, "dev": round(fma - 25.0, 2), "status": "Hyperdivergent" if fma > 30 else ("Hypodivergent" if fma < 20 else "Normal")},
            {"param": "Co-A (Maxilla Length)", "measured": co_a, "unit": "mm", "norm": 90.0, "dev": round(co_a - 90.0, 2), "status": "Increased" if co_a > 95 else ("Decreased" if co_a < 85 else "Normal")},
            {"param": "Co-Gn (Mandible Length)", "measured": co_gn, "unit": "mm", "norm": 115.0, "dev": round(co_gn - 115.0, 2), "status": "Increased" if co_gn > 122 else ("Decreased" if co_gn < 108 else "Normal")},
            {"param": "Upper Lip to E-Line", "measured": dist_ls, "unit": "mm", "norm": -4.0, "dev": round(dist_ls - (-4.0), 2), "status": "Protrusive" if dist_ls > -2 else ("Retrusive" if dist_ls < -6 else "Normal")},
            {"param": "Lower Lip to E-Line", "measured": dist_li, "unit": "mm", "norm": -2.0, "dev": round(dist_li - (-2.0), 2), "status": "Protrusive" if dist_li > 0 else ("Retrusive" if dist_li < -4 else "Normal")}
        ]

        # --- ۹. تفسیر جامع و تخصصی داده‌ها ---
        detailed_interpretations = {
            "رابطه اسکلتی ساژیتال (Sagittal Relationship)": f"مقدار زاویه ANB برابر با {anb}° و ارزیابی Wits برابر با {wits_mm} mm می‌باشد. " + 
                ("نشان‌دهنده ناهنجاری اسکلتی کلاس II شدید به دلیل برآمدگی فک بالا یا عقب‌ماندگی فک پایین است." if anb > 4.5 else 
                 ("نشان‌دهنده ناهنجاری اسکلتی کلاس III به دلیل جلو بودن فک پایین یا ضعیف بودن فک بالا است." if anb < 0.5 else 
                  "رابطه فک بالا و پایین در حد فاصل نرمال است و تطابق اسکلتی کلاس I وجود دارد.")),

            "الگوی رشد عمودی (Vertical Pattern)": f"زاویه FMA برابر با {fma}° است (Norm: 25.0°). " + 
                ("الگوی رشد هایپردایورجنت (Vertical Growing / High Angle). بیمار تمایل به اوپن بایت، افزایش ارتفاع تحتانی صورت و عضلات ضعیف‌تر جویدن دارد." if fma > 30 else 
                 ("الگوی رشد هایپودایورجنت (Horizontal Growing / Low Angle). بیمار دارای ساختار صورت فشرده، تمایل به دیپ بایت و عضلات جویدن قوی است." if fma < 20 else 
                  "الگوی رشد عمودی متوازن و نرمال (Mesofacial/Normodivergent).")),

            "تحلیل تناسب طول فکین (McNamara Discrepancy)": f"طول موثر فک بالا (Co-A) برابر {co_a} mm و طول موثر فک پایین (Co-Gn) برابر {co_gn} mm است (اختلاف: {diff_mcnamara} mm). " +
                ("اختلاف طول فکین بیشتر از حد نرمال بوده که موید رشد بیش از حد فک پایین یا کوتاهی فک بالا می‌باشد." if diff_mcnamara > 28 else
                 ("اختلاف طول فکین کمتر از حد نرمال بوده که نشان‌دهنده نقص رشد فک پایین است." if diff_mcnamara < 20 else
                  "تناسب طولی فک بالا و پایین در محدوده هارمونیک قرار دارد.")),

            "پروفایل بافت نرم (Soft Tissue Profile)": f"فاصله لب بالا تا خط E برابر با {dist_ls} mm و لب پایین {dist_li} mm است. " +
                ("برجستگی بافت نرم لب‌ها نسبت به خط E مشهود است که می‌تواند ناشی از پروتروژن دندان‌های قدامی (Bimaxillary Protrusion) باشد." if dist_ls > -1 or dist_li > 1 else
                 ("پروفایل لب‌ها نسبت به خط E عقب‌رفته (Retrusive) است." if dist_ls < -6 else
                  "پروفایل بافت نرم و وضعیت لب‌ها زیبا و متوازن است."))
        }

        # طرح درمان پیشنهادی
        treatment_plan = []
        if anb > 4:
            if fma > 30:
                treatment_plan.append({"title": "کنترل رشد عمودی و اصلاح کلاس II", "desc": "استفاده از دستگاه‌های اینترودکننده یا Tilted Occlusal Plane control همراه با الستیک‌های کلاس II جهت جلوگیری از چرخش ساعت‌گرد فک پایین."})
            else:
                treatment_plan.append({"title": "تحریک رشد/پیش‌آوردن فک پایین", "desc": "دستگاه‌های فانکشنال (مانند Twin Block یا Herbst) در سنین رشد، یا جراحی BSSO در سنین بالاتر."})
        elif anb < 0:
            treatment_plan.append({"title": "پروتراکشن فک بالا یا اصلاح کلاس III", "desc": "فیس‌ماسک در سنین رشد یا جراحی دو فک در سنین بالاتر."})
        else:
            treatment_plan.append({"title": "ارتودنسی مرتب‌سازی دندانی", "desc": "تمرکز بر ردیف‌سازی دندان‌ها، اصلاح Crowding و تنظیم قوس‌ها بدون مداخله اسکلتی."})

        if dist_ls > 0 or dist_li > 1:
            treatment_plan.append({"title": "ارزیابی طرح درمان کشیدن دندان (Extraction Evaluation)", "desc": "به دلیل برجستگی لب‌ها نسبت به خط E، بررسی کشیدن پری‌مولرها جهت ریترود کردن دندان‌های قدامی پیشنهاد می‌شود."})

        # --- ۱۰. نمایش در UI Streamlit ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Steiner (ANB)", f"{anb}°", f"{round(anb - 2.0, 2)}°")
        m2.metric("Wits", f"{wits_mm} mm", f"{round(wits_mm - wits_norm, 2)} mm")
        m3.metric("McNamara Diff", f"{diff_mcnamara} mm")
        m4.metric("Downs (FMA)", f"{fma}°", f"{round(fma - 25.0, 2)}°")

        st.divider()
        st.header("📊 جدول مقایسه کامل با Normها و آنالیز جامع")
        
        tab1, tab2, tab3 = st.tabs(["📐 جدول مقایسه با Normها", "🔍 تفسیر تخصصی داده‌ها", "💡 طرح درمان پیشنهادی"])
        
        with tab1:
            st.subheader("مقایسه اندازه پارامترها با مقادیر مرجع (Norms)")
            st.dataframe(norm_table_data, use_container_width=True)

        with tab2:
            st.subheader("تحلیل تفصیلی ناهنجاری‌ها")
            for cat, desc in detailed_interpretations.items():
                st.markdown(f"**• {cat}:**")
                st.write(desc)

        with tab3:
            st.subheader("دستورالعمل‌های پیشنهادی درمان")
            for idx, item in enumerate(treatment_plan, 1):
                st.markdown(f"**{idx}. {item['title']}**")
                st.write(item['desc'])

        # --- ۱۱. تولید و دکمه دانلود PDF ---
        st.markdown("---")
        
        patient_info_pdf = {
            'gender': gender,
            'pixel_size': pixel_size,
            'date': '2026-09-10',
            'diag': "Class II Skeletal" if anb > 4 else ("Class III Skeletal" if anb < 0 else "Class I Skeletal")
        }
        
        img_byte_arr = io.BytesIO()
        draw_img.save(img_byte_arr, format='PNG')
        annotated_img_bytes = img_byte_arr.getvalue()
        
        pdf_bytes = generate_clinical_pdf(patient_info_pdf, norm_table_data, detailed_interpretations, treatment_plan, annotated_img_bytes)
        
                st.download_button(
            label="📄 دانلود گزارش جامع چندصفحه‌ای بالینی و Norms (PDF)",
            data=pdf_bytes,
            file_name=f"Aariz_Comprehensive_Report_{uploaded_file.name.split('.')[0]}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        # --- دانلود نتایج به صورت JSON برای ادغام با تحلیل ۳D ---
        import json
        from datetime import datetime

        ceph_results_json = {
            "version": "1.0",
            "type": "cephalometric_analysis",
            "timestamp": datetime.now().isoformat(),
            "patient_info": {
                "gender": gender,
                "pixel_size": pixel_size,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "diagnosis": patient_info_pdf['diag']
            },
            "measurements": {
                "SNA": sna,
                "SNB": snb,
                "ANB": anb,
                "Wits": wits_mm,
                "FMA": fma,
                "Co_A": co_a,
                "Co_Gn": co_gn,
                "McNamara_Diff": diff_mcnamara,
                "Upper_Lip_E_Line": dist_ls,
                "Lower_Lip_E_Line": dist_li
            },
            "norms_table": norm_table_data,
            "interpretations": detailed_interpretations,
            "treatment_plan": treatment_plan
        }

        st.download_button(
            label="💾 دانلود نتایج ۲D (JSON) - برای ادغام با تحلیل ۳D",
            data=json.dumps(ceph_results_json, ensure_ascii=False, indent=2),
            file_name=f"ceph_results_{uploaded_file.name.split('.')[0]}.json",
            mime="application/json",
            use_container_width=True
        )

        gc.collect()

# --- ۱۲. فراخوانی تب اسکن داخل دهانی ۳D ---
with tab_3d:
    if render_intraoral_3d_tab is not None:
        render_intraoral_3d_tab()
    else:
        st.warning("⚠️ ماژول `intraoral_3d_module.py` در کنار فایل اصلی یافت نشد. لطفاً این فایل را در مسیر برنامه قرار دهید.")
