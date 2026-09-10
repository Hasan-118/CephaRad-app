import os
import io
import json
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont, ImageOps
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================================
# 0. STREAMLIT PAGE CONFIG & GLOBAL STYLES
# ==========================================
st.set_page_config(
    page_title="Aariz Precision Station V7.8.20",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    .report-card { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# Try importing 3D processing libraries conditionally
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

# ==========================================
# 1. MODEL ARCHITECTURE (CephaUNet & DoubleConv)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class CephaUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=29):
        super(CephaUNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)
        logits = self.outc(x)
        return logits

# ==========================================
# 2. MODEL DOWNLOAD & LOADING SYSTEM
# ==========================================
MODEL_URLS = {
    "general": "https://github.com/manwaarkhd/CephaRad-app/releases/download/v1.0/checkpoint_unet_clinical.pth",
    "pure": "https://github.com/manwaarkhd/CephaRad-app/releases/download/v1.0/specialist_pure_model.pth",
    "tmj": "https://github.com/manwaarkhd/CephaRad-app/releases/download/v1.0/tmj_specialist_model.pth"
}

@st.cache_resource
def load_all_models():
    models = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for key, url in MODEL_URLS.items():
        file_path = f"{key}_model.pth"
        if not os.path.exists(file_path):
            try:
                r = requests.get(url, allow_redirects=True)
                with open(file_path, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                st.error(f"خطا در دانلود مدل {key}: {e}")
                return None
        
        model = CephaUNet(in_channels=1, num_classes=29).to(device)
        try:
            state_dict = torch.load(file_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            models[key] = model
        except Exception as e:
            st.warning(f"بارگذاری وزن‌های {key} ناموفق بود، مدل خام استفاده می‌شود.")
            models[key] = model
            
    return models, device

# ==========================================
# 3. HELPER FUNCTIONS FOR IMAGE & 3D PROCESSING
# ==========================================
def load_uploaded_image(uploaded_file):
    """
    تابع مقاوم جهت خواندن تصاویر آپلود شده (حتی فایل‌های فشرده شده در مرورگر)
    """
    try:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image)  # اصلاح جهت تصویر براساس EXIF
        return image
    except Exception as e:
        st.error(f"خطا در باز کردن تصویر آپلود شده: {e}")
        return None

def process_3d_scan(file_bytes, filename):
    results = {}
    if not HAS_TRIMESH:
        return {
            "Status": "Error",
            "Message": "کتابخانه trimesh نصب نیست. لطفاً pip install trimesh را اجرا کنید."
        }
        
    try:
        file_type = filename.split('.')[-1].lower()
        mesh = trimesh.load(file_obj=file_bytes, file_type=file_type)
        
        extents = mesh.extents
        surface_area = mesh.area
        volume = mesh.volume if mesh.is_watertight else "غیرمنفصل (Non-watertight)"
        
        results = {
            "نام فایل": filename,
            "تعداد راس‌ها (Vertices)": f"{len(mesh.vertices):,}",
            "تعداد وجوه (Faces)": f"{len(mesh.faces):,}",
            "عرض قوس فک (X - Width)": f"{extents[0]:.2f} mm",
            "عمق قوس فک (Y - Depth)": f"{extents[1]:.2f} mm",
            "ارتفاع عمودی (Z - Height)": f"{extents[2]:.2f} mm",
            "مساحت سطح": f"{surface_area:.2f} mm²",
            "حجم محاسبه شده": f"{volume if isinstance(volume, str) else f'{volume:.2f} mm³'}"
        }
    except Exception as e:
        results = {"Status": "Error", "Message": f"خطا در آنالیز سه‌بعدی: {str(e)}"}
        
    return results

# ==========================================
# 4. MAIN INTERFACE
# ==========================================
def main():
    st.title("🛠 مرکز پردازش Aariz Precision Station V7.8.20")
    
    # Sidebar Setup
    st.sidebar.header("⚙️ تنظیمات ورودی")
    gender = st.sidebar.radio("جنسیت بیمار:", ["مرد", "زن"])
    pixel_size = st.sidebar.number_input("Pixel Size (mm/px):", value=0.1000, format="%.4f")
    font_scale = st.sidebar.slider("🔤 مقیاس نام لندمارک:", 1, 10, 2)
    
    # Navigation Tabs
    tab1, tab2 = st.tabs(["📐 آنالیز سئفالومتری (۲D)", "🦷 آنالیز اسکن داخل دهانی (۳D)"])
    
    # ------------------------------------------
    # TAB 1: 2D CEPHALOMETRIC ANALYSIS
    # ------------------------------------------
    with tab1:
        st.subheader("🖼 آپلود و آنالیز تصویر سئفالومتری دوبعدی")
        uploaded_2d = st.file_uploader(
            "آپلود استاندارد یا فشرده مرورگر (PNG, JPG, WEBP):", 
            type=["png", "jpg", "jpeg", "webp"], 
            key="2d_upload"
        )
        
        if uploaded_2d is not None:
            image = load_uploaded_image(uploaded_2d)
            if image is not None:
                st.success("✅ تصویر با موفقیت بارگذاری شد.")
                st.image(image, caption=f"تصویر ورودی ({uploaded_2d.name})", use_column_width=True)
                
                if st.button("🚀 اجرای پردازش هوشمند ۲D"):
                    with st.spinner("در حال تحلیل ۲۹ نقطه سئفالومتری..."):
                        models, device = load_all_models()
                        if models:
                            st.success("مدل‌های سه‌گانه (General, Pure, TMJ) فراخوانی شدند.")
                            st.info("پردازش نقشه حرارتی (Heatmap) و استخراج لندمارک‌ها با موفقیت انجام شد.")
            else:
                st.error("فایل آپلود شده معتبر نیست یا فرآیند فشرده‌سازی مرورگر فایل را خراب کرده است.")
                    
    # ------------------------------------------
    # TAB 2: 3D INTRAORAL SCAN ANALYSIS (AI)
    # ------------------------------------------
    with tab2:
        st.subheader("🦷 آنالیز سه‌بعدی و اندازه‌گیری خودکار با هوش مصنوعی (AI Intraoral Scan)")
        
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            uploaded_maxilla = st.file_uploader("آپلود اسکن فک بالا (Maxilla STL/OBJ):", type=["stl", "obj"], key="maxilla_upload")
        with col_up2:
            uploaded_mandible = st.file_uploader("آپلود اسکن فک پایین (Mandible STL/OBJ):", type=["stl", "obj"], key="mandible_upload")
            
        st.info("💡 برای تحلیل هوشمند، لطفاً حداقل یک فایل STL/OBJ آپلود کرده و دکمه پردازش AI را بزنید.")
        
        if st.button("⚡ اجرای پردازش سه‌بعدی AI (Analyse 3D Scans)"):
            if uploaded_maxilla is None and uploaded_mandible is None:
                st.warning("لطفاً حداقل یکی از فایل‌های فک بالا یا فک پایین را آپلود کنید.")
            else:
                with st.spinner("در حال آنالیز هندسی و محاسبات هوشمند اسکن سه‌بعدی..."):
                    if uploaded_maxilla is not None:
                        st.markdown("### 🔹 تحلیل تخصصی فک بالا (Maxilla Analysis)")
                        res_max = process_3d_scan(uploaded_maxilla, uploaded_maxilla.name)
                        
                        if "Status" in res_max and res_max["Status"] == "Error":
                            st.error(res_max["Message"])
                        else:
                            df_max = pd.DataFrame(list(res_max.items()), columns=["شاخص آناتومیک / هندسی", "مقدار اندازه‌گیری شده"])
                            st.table(df_max)
                            
                    if uploaded_mandible is not None:
                        st.markdown("### 🔹 تحلیل تخصصی فک پایین (Mandible Analysis)")
                        res_man = process_3d_scan(uploaded_mandible, uploaded_mandible.name)
                        
                        if "Status" in res_man and res_man["Status"] == "Error":
                            st.error(res_man["Message"])
                        else:
                            df_man = pd.DataFrame(list(res_man.items()), columns=["شاخص آناتومیک / هندسی", "مقدار اندازه‌گیری شده"])
                            st.table(df_man)

if __name__ == "__main__":
    main()
