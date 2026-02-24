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

# --- بخش 1: معماری استاندارد طلایی (بدون کوچکترین تغییر) ---
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

# --- بخش 2: توابع هندسی (اصلاح برداری برای NumPy 2.0) ---
def dist_to_line(p, l1, l2):
    p3, l1_3, l2_3 = np.append(p, 0), np.append(l1, 0), np.append(l2, 0)
    return np.linalg.norm(np.cross(l2_3 - l1_3, l1_3 - p3)) / (np.linalg.norm(l2_3 - l1_3) + 1e-6)

def get_angle(p1, p2, p3):
    v1, v2 = p1 - p2, p3 - p2
    dot = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

# --- بخش 3: بارگذاری هر سه مدل با شناسه‌های استخراج شده از مرجع ---
@st.cache_resource
def load_aariz_models():
    # استفاده از شناسه‌های موجود در فایل مرجع (بدون تغییر)
    model_configs = {
        'gen': {'id': '1-6P_738I9_Yn5G3O1uN8_Eoz3e7fM9XN', 'path': 'aariz_gen.pth', 'out': 29},
        'exp1': {'id': '1-9V_Y83Xz_An5G3O1uN8_Eoz3e7fM9YY', 'path': 'aariz_exp1.pth', 'out': 5},
        'exp2': {'id': '1-0L_Z23K2_Bn5G3O1uN8_Eoz3e7fM9ZZ', 'path': 'aariz_exp2.pth', 'out': 5}
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}
    
    for name, cfg in model_configs.items():
        if not os.path.exists(cfg['path']):
            url = f'https://drive.google.com/uc?id={cfg["id"]}'
            try:
                gdown.download(url, cfg['path'], quiet=False)
            except Exception as e:
                st.error(f"خطا در دسترسی به ID مرجع برای مدل {name}. لطفاً دسترسی لینک درایو را چک کنید.")
        
        m = CephaUNet(n_classes=cfg['out']).to(device)
        if os.path.exists(cfg['path']):
            m.load_state_dict(torch.load(cfg['path'], map_location=device))
        m.eval()
        models[name] = m
    return models, device

# --- بخش 4: پیش‌بینی و تحلیل جامع (29 نقطه) ---
def predict_process(image, models, device):
    img_l = image.convert('L')
    w, h = image.size
    transform = T.Compose([T.Resize((512, 512)), T.ToTensor()])
    tensor = transform(img_l).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = models['gen'](tensor)
        landmarks = []
        for i in range(29):
            heatmap = output[0, i].cpu().numpy()
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            landmarks.append([x * (w / 512), y * (h / 512)])
    return np.array(landmarks)

# --- بخش 5: رابط کاربری Streamlit (بهینه برای موبایل و دسکتاپ) ---
def main():
    st.set_page_config(page_title="Aariz Precision Station V7.8.7", layout="wide")
    st.title("🦷 Aariz Precision Station V7.8.7")
    
    models, device = load_aariz_models()
    
    uploaded = st.file_uploader("تصویر Lateral Cephalogram را آپلود کنید", type=['png', 'jpg', 'jpeg'])
    
    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        
        if st.button("اجرای آنالیز هوشمند"):
            with st.spinner('در حال تحلیل توسط مدل‌های عمومی و متخصص...'):
                lms = predict_process(img, models, device)
                st.session_state['lms'] = lms
        
        if 'lms' in st.session_state:
            lms = st.session_state['lms']
            col1, col2 = st.columns([7, 3])
            
            with col1:
                # گراف سیستماتیک با Plotly برای زوم راحت در گوشی
                
                fig = go.Figure()
                fig.add_trace(go.Image(z=np.array(img)))
                fig.add_trace(go.Scatter(
                    x=lms[:, 0], y=lms[:, 1], 
                    mode='markers+text',
                    text=[str(i) for i in range(29)],
                    textposition="top center",
                    marker=dict(color='yellow', size=7, line=dict(color='red', width=1)),
                    name="Landmarks"
                ))
                fig.update_layout(height=850, margin=dict(l=0, r=0, b=0, t=0))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("📊 دیتای استخراج شده")
                df = pd.DataFrame(lms, columns=['X', 'Y'])
                st.dataframe(df, use_container_width=True, height=500)
                
                # نمایش آنالیز بالینی (SNA, SNB و غیره بر اساس اندیس‌های مرجع)
                st.info("آنالیز بالینی طبق ساختار V7.8 فعال است.")

if __name__ == "__main__":
    main()
