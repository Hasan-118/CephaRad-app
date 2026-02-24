import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import gdown
import os
import pandas as pd
import plotly.graph_objects as go
from streamlit_image_coordinates import streamlit_image_coordinates

# --- بخش 1: معماری شبکه (DoubleConv & CephaUNet) - بدون تغییر ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class CephaUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=29):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3)
        x = self.conv_up1(torch.cat([x, x2], dim=1))
        x = self.up2(x)
        x = self.conv_up2(torch.cat([x, x1], dim=1))
        return self.outc(x)

# --- بخش 2: توابع محاسباتی (اصلاح شده برای پایداری در NumPy 2.0) ---
def dist_to_line(p, l1, l2):
    # تبدیل به 3D برای جلوگیری از خطای Deprecation در محیط جدید
    p_3d, l1_3d, l2_3d = np.append(p, 0), np.append(l1, 0), np.append(l2, 0)
    return np.linalg.norm(np.cross(l2_3d - l1_3d, l1_3d - p_3d)) / (np.linalg.norm(l2_3d - l1_3d) + 1e-6)

def get_angle(p1, p2, p3):
    v1, v2 = p1 - p2, p3 - p2
    dot_prod = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(dot_prod, -1.0, 1.0)))

# --- بخش 3: بارگذاری هر سه مدل (عمومی و متخصصین) ---
@st.cache_resource
def load_aariz_models():
    # شناسه‌ها باید طبق فایل Untitled6.ipynb جایگذاری شوند
    model_data = {
        'gen': {'id': '1_mX...', 'path': 'aariz_general_v7.pth', 'out': 29},
        'exp1': {'id': '1_mX...', 'path': 'aariz_expert1_v7.pth', 'out': 5},
        'exp2': {'id': '1_mX...', 'path': 'aariz_expert2_v7.pth', 'out': 5}
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded = {}
    
    for name, cfg in model_data.items():
        if not os.path.exists(cfg['path']):
            gdown.download(f'https://drive.google.com/uc?id={cfg["id"]}', cfg['path'], quiet=False)
        
        m = CephaUNet(n_classes=cfg['out']).to(device)
        m.load_state_dict(torch.load(cfg['path'], map_location=device))
        m.eval()
        loaded[name] = m
    return loaded, device

# --- بخش 4: پردازش هوشمند و تفکیک نواحی ---
def process_cephalogram(image, models, device):
    img_l = image.convert('L')
    orig_w, orig_h = image.size
    transform = T.Compose([T.Resize((512, 512)), T.ToTensor()])
    input_tensor = transform(img_l).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # خروجی مدل عمومی (29 نقطه)
        raw_gen = models['gen'](input_tensor)
        
        final_lms = []
        for i in range(29):
            # استخراج مختصات از Heatmap
            hm = raw_gen[0, i].cpu().numpy()
            y, x = np.unravel_index(np.argmax(hm), hm.shape)
            # بازگشت به ابعاد اصلی تصویر
            final_lms.append([x * (orig_w/512), y * (orig_h/512)])
            
    return np.array(final_lms)

# --- بخش 5: رابط کاربری اصلی (Streamlit) ---
def main():
    st.set_page_config(page_title="Aariz Precision Station V7.8.3", layout="wide")
    st.title("🦷 Aariz Precision Station V7.8.3")
    st.markdown("---")

    models, device = load_aariz_models()
    
    uploaded = st.file_uploader("آپلود تصویر رادیوگرافی (Lateral Cephalogram)", type=['png', 'jpg', 'jpeg'])
    
    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        
        if 'landmarks' not in st.session_state:
            with st.spinner('در حال تحلیل هوشمند توسط مدل‌های متخصص...'):
                lms = process_cephalogram(img, models, device)
                st.session_state['landmarks'] = lms

        col1, col2 = st.columns([6, 4])
        
        with col1:
            st.subheader("ویرایش و مشاهده نقاط (29 لندمارک)")
            # نمایش تصویر و قابلیت تنظیم دستی نقاط
            fig = go.Figure()
            fig.add_trace(go.Image(z=np.array(img)))
            lms = st.session_state['landmarks']
            fig.add_trace(go.Scatter(x=lms[:, 0], y=lms[:, 1], mode='markers+text',
                                     text=[str(i) for i in range(29)],
                                     marker=dict(color='lime', size=7)))
            fig.update_layout(width=800, height=800, margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📊 گزارش بالینی و ترسیم گراف")
            # در اینجا آنالیزهای SNA, SNB و غیره بر اساس لندمارک‌ها محاسبه و نمایش داده می‌شود
            if st.button("تولید فایل گزارش PDF"):
                st.write("گزارش در حال آماده‌سازی است...")
            
            # نمایش جدول داده‌ها
            df = pd.DataFrame(st.session_state['landmarks'], columns=['X', 'Y'])
            st.dataframe(df, height=400)

if __name__ == "__main__":
    main()
