                                                                                                                                                                       import streamlit as st

import torch

import torch.nn as nn

import numpy as np

import os

import json

from datetime import datetime

from PIL import Image, ImageDraw

import torchvision.transforms as transforms

from streamlit_image_coordinates import streamlit_image_coordinates

import math



# --- ۱. تنظیمات اولیه و پوشه ذخیره‌سازی ---

RESULTS_DIR = "Aariz_Results"

if not os.path.exists(RESULTS_DIR):

    os.makedirs(RESULTS_DIR)



# --- ۲. تعریف معماری مدل (بدون تغییر) ---

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



# --- ۳. بارگذاری مدل‌ها (بهینه شده) ---

@st.cache_resource

def load_aariz_models():

    current_dir = os.path.dirname(os.path.abspath(__file__))

    model_files = [

        'checkpoint_unet_clinical.pth',

        'specialist_pure_model.pth',

        'tmj_specialist_model.pth'

    ]

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded_models = []

    

    for f in model_files:

        full_path = os.path.join(current_dir, f)

        if os.path.exists(full_path):

            try:

                m = CephaUNet(n_landmarks=29).to(device)

                ckpt = torch.load(full_path, map_location=device)

                state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

                m.load_state_dict(state_dict)

                m.eval()

                # بهینه‌سازی سرعت اگر GPU در دسترس باشد

                if device.type == 'cuda': m = m.half()

                loaded_models.append(m)

            except Exception as e:

                st.sidebar.error(f"خطا در لود {f}: {e}")

    return loaded_models, device



# --- ۴. پیش‌بینی فوق سریع (Inference Mode) ---

def run_ai_prediction(img_path, models, device):

    img_orig = Image.open(img_path).convert('L')

    orig_size = img_orig.size

    img_resized = img_orig.resize((512, 512), Image.LANCZOS)

    

    input_tensor = transforms.ToTensor()(img_resized).unsqueeze(0).to(device)

    if device.type == 'cuda': input_tensor = input_tensor.half()

    

    with torch.inference_mode(): # سریع‌تر از no_grad

        outs = [mod(input_tensor)[0].cpu().float().numpy() for mod in models]

    

    ANT_IDX = [10, 14, 9, 5, 28, 20]

    POST_IDX = [7, 11, 12, 15]

    

    coords = {}

    sx, sy = orig_size[0]/512, orig_size[1]/512

    num_m = len(outs)



    for i in range(29):

        # سیستم هوشمند Ensemble

        if i in ANT_IDX and num_m >= 2: hm = outs[1][i]

        elif i in POST_IDX and num_m >= 3: hm = outs[2][i]

        else: hm = outs[0][i]

            

        y, x = np.unravel_index(np.argmax(hm), hm.shape)

        coords[i] = [int(x * sx), int(y * sy)]

    return coords



# --- ۵. رابط کاربری اصلی ---

st.set_page_config(page_title="Aariz Station V2", layout="wide")

models, device = load_aariz_models()



# سایدبار وضعیت سخت‌افزار

st.sidebar.title("⚙️ سیستم و سخت‌افزار")

st.sidebar.write(f"🖥️ **Device:** `{device.type.upper()}`")

st.sidebar.write(f"📦 **مدل‌های فعال:** `{len(models)}/3`")



if not models:

    st.error("فایل‌های وزن مدل پیدا نشدند.")

    st.stop()



landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']

weak_landmarks = [9, 14, 16, 18, 19, 22, 23]



st.sidebar.title("🧠 Aariz AI Control")

base_dir = st.sidebar.text_input("مسیر پروژه:", value=os.getcwd())

img_folder = os.path.join(base_dir, "Aariz", "train", "Cephalograms")



if os.path.exists(img_folder):

    files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg'))]

    selected_file = st.sidebar.selectbox("انتخاب سفالوگرام:", files)

    target_idx = st.sidebar.selectbox("نقطه برای بازبینی:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")

    

    img_path = os.path.join(img_folder, selected_file)

    

    # مدیریت حافظه برای سرعت لود تصویر

    if "current_img" not in st.session_state or st.session_state.current_img != selected_file:

        with st.spinner('در حال پردازش هوشمند...'):

            st.session_state.lms = run_ai_prediction(img_path, models, device)

            st.session_state.current_img = selected_file



    col1, col2 = st.columns([2, 1])

    

    with col1:

        raw_img = Image.open(img_path).convert("RGB")

        draw_img = raw_img.copy()

        draw = ImageDraw.Draw(draw_img)

        l = st.session_state.lms

        

        # ترسیم خطوط Steiner

        draw.line([tuple(l[10]), tuple(l[4]), tuple(l[0])], fill="yellow", width=4)

        draw.line([tuple(l[4]), tuple(l[2])], fill="cyan", width=4)

        

        for i, pos in l.items():

            c = "red" if i == target_idx else ("orange" if i in weak_landmarks else "#00FF00")

            r = 15 if i == target_idx else 7

            draw.ellipse([pos[0]-r, pos[1]-r, pos[0]+r, pos[1]+r], fill=c)



        st.subheader(f"📍 در حال تنظیم: {landmark_names[target_idx]}")

        # تعامل سریع با کلیک

        res = streamlit_image_coordinates(draw_img, width=800, key="aariz_v2")

        

        if res:

            scale = raw_img.width / 800

            nx, ny = int(res["x"]*scale), int(res["y"]*scale)

            if l[target_idx] != [nx, ny]:

                st.session_state.lms[target_idx] = [nx, ny]

                st.rerun()



    with col2:

        st.header("📊 Clinical Report")

        def angle(p1, p2, p3):

            v1, v2 = np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)

            return round(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1, 1))), 2)

        

        sna = angle(l[10], l[4], l[0])

        snb = angle(l[10], l[4], l[2])

        anb = round(sna - snb, 2)

        

        st.metric("SNA", f"{sna}°")

        st.metric("SNB", f"{snb}°")

        st.metric("ANB", f"{anb}°", delta="Class II" if anb > 4 else ("Class III" if anb < 0 else "Class I"))



        # --- دکمه ذخیره واقعی در دیتابیس ---

        if st.button("💾 ذخیره نهایی و ثبت در درایو"):

            p_folder = os.path.join(RESULTS_DIR, selected_file.split('.')[0])

            if not os.path.exists(p_folder): os.makedirs(p_folder)

            

            # ذخیره JSON

            data = {

                "patient": selected_file,

                "timestamp": datetime.now().isoformat(),

                "landmarks": st.session_state.lms,

                "measurements": {"SNA": sna, "SNB": snb, "ANB": anb}

            }

            with open(os.path.join(p_folder, "data.json"), "w") as f:

                json.dump(data, f, indent=4)

            

            # ذخیره عکس آنالیز شده

            draw_img.save(os.path.join(p_folder, "analysis.png"))

            

            st.success(f"✅ با موفقیت در پوشه {p_folder} ذخیره شد.")

            st.balloons()
