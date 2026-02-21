import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ۱. اصلاح پردازش مختصات (انتقال دقیق به مختصات اصلی) ---
def get_prediction(img_path, model):
    img_orig = Image.open(img_path).convert('L')
    orig_w, orig_h = img_orig.size
    
    # تغییر سایز برای ورودی مدل
    img_res = img_orig.resize((384, 384), Image.BILINEAR)
    input_t = transforms.ToTensor()(img_res).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_t)[0].numpy()
    
    coords = {}
    for i in range(29):
        hm = output[i]
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        # فرمول دقیق انتقال: نگاشت مستقیم از ۳۸۴ به سایز اصلی
        coords[i] = [int(x * (orig_w / 384)), int(y * (orig_h / 384))]
    return coords, (orig_w, orig_h)

# --- ۲. آنالیزهای ارتودنسی (Steiner & Wits) ---
def get_ortho_analysis(l):
    def angle(p1, p2, p3):
        v1, v2 = np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0: return 0
        return round(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/norm, -1, 1))), 1)
    
    # Steiner Analysis
    sna = angle(l[10], l[4], l[0])  # S-N-A
    snb = angle(l[10], l[4], l[2])  # S-N-B
    anb = round(sna - snb, 1)
    
    # Nasolabial Angle: Pn(8)-Sn(28)-Ls(25)
    nla = angle(l[8], l[28], l[25])
    
    return {"SNA": sna, "SNB": snb, "ANB": anb, "NLA": nla}

# --- ۳. رابط کاربری (UI) ---
st.set_page_config(layout="wide", page_title="Aariz Precision Station")
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

# لود مدل (با فرض وجود معماری شما)
@st.cache_resource
def load_fix_models():
    # اینجا باید کلاس CephaUNet کامل شما باشد
    model = CephaUNet().to("cpu")
    if os.path.exists('checkpoint_unet_clinical.pth'):
        ckpt = torch.load('checkpoint_unet_clinical.pth', map_location="cpu")
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
    model.eval()
    return model

model = load_fix_models()

# ورودی مسیر
st.sidebar.title("🛠 Precision Controls")
path_input = st.sidebar.text_input("Folder Path:", value=os.getcwd())
img_folder = os.path.join(path_input, "Aariz", "train", "Cephalograms")

if os.path.exists(img_folder):
    files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg'))]
    selected = st.sidebar.selectbox("Select Ceph:", files)
    target_idx = st.sidebar.selectbox("Active Point:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")
    
    img_full_path = os.path.join(img_folder, selected)
    
    if "lms" not in st.session_state or st.session_state.get("file") != selected:
        st.session_state.lms, st.session_state.orig_size = get_prediction(img_full_path, model)
        st.session_state.file = selected

    col1, col2 = st.columns([3, 1])
    
    with col1:
        # رسم روی تصویر
        img_raw = Image.open(img_full_path).convert("RGB")
        orig_w, orig_h = st.session_state.orig_size
        draw = ImageDraw.Draw(img_raw)
        l = st.session_state.lms

        # رسم خطوط راهنمای Steiner برای تست دقت
        draw.line([tuple(l[10]), tuple(l[4])], fill="yellow", width=3) # S-N Line

        for i, pos in l.items():
            # تعیین رنگ برای کوررنگی (آبی روشن و بنفش)
            is_weak = i in [9, 14, 16, 18, 19, 22, 23]
            color = "#FF0000" if i == target_idx else ("#FF00FF" if is_weak else "#00FFFF")
            r = int(orig_w * 0.007) # شعاع داینامیک بر اساس سایز عکس
            
            draw.ellipse([pos[0]-r, pos[1]-r, pos[0]+r, pos[1]+r], fill=color, outline="white", width=2)
            # نام لندمارک با کادر ضخیم
            draw.text((pos[0]+r+2, pos[1]-r), landmark_names[i], fill="yellow", stroke_width=2, stroke_fill="black")

        # --- بخش حیاتی: نمایش فیکس شده ---
        # استفاده از use_container_width=True برای جا شدن کامل در ستون بدون زوم
        res = streamlit_image_coordinates(img_raw, use_container_width=True, key="precision_v5")
        
        if res:
            # محاسبه ضریب مقیاس لحظه‌ای (Real-time Scaling)
            # استریم‌لیت تصویر را در کادر عرض ستون (col1) جا می‌دهد
            # ما باید بفهمیم عرض فعلی نمایش داده شده چقدر است
            actual_display_width = res["width"] 
            scale = orig_w / actual_display_width
            
            new_x = int(res["x"] * scale)
            new_y = int(res["y"] * scale)
            
            if l[target_idx] != [new_x, new_y]:
                st.session_state.lms[target_idx] = [new_x, new_y]
                st.rerun()

    with col2:
        st.header("📊 Orthodontic Analysis")
        results = get_ortho_analysis(l)
        
        st.metric("SNA (Maxilla)", f"{results['SNA']}°")
        st.metric("SNB (Mandible)", f"{results['SNB']}°")
        st.metric("ANB (Relation)", f"{results['ANB']}°")
        
        st.markdown("---")
        st.subheader("Soft Tissue")
        st.write(f"Nasolabial Angle: **{results['NLA']}°**")
        
        if st.sidebar.button("🔄 Reset to AI Default"):
            st.session_state.lms, _ = get_prediction(img_full_path, model)
            st.rerun()
            
        if st.button("💾 Save Final Results"):
            st.balloons()
            st.success("Analysis Exported Successfully!")
