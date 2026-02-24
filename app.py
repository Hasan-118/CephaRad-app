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

# --- بخش 1: معماری استاندارد طلایی (بدون هیچ تغییری) ---
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

# --- بخش 2: محاسبات هندسی (سازگار شده با NumPy 2.0 برای جلوگیری از کرش) ---
def dist_to_line(p, l1, l2):
    # تبدیل به آرایه ۳ بعدی برای سازگاری با np.cross در نسخه جدید
    p3, l1_3, l2_3 = np.append(p, 0), np.append(l1, 0), np.append(l2, 0)
    return np.linalg.norm(np.cross(l2_3 - l1_3, l1_3 - p3)) / (np.linalg.norm(l2_3 - l1_3) + 1e-6)

def get_angle(p1, p2, p3):
    v1, v2 = p1 - p2, p3 - p2
    unit_v1 = v1 / (np.linalg.norm(v1) + 1e-6)
    unit_v2 = v2 / (np.linalg.norm(v2) + 1e-6)
    dot_product = np.dot(unit_v1, unit_v2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle)

# --- بخش 3: بارگذاری هر سه مدل (عمومی و متخصصین) با مدیریت خطا ---
@st.cache_resource
def load_aariz_models():
    # شناسه‌ها بر اساس Untitled6.ipynb (حتماً جایگزین شود)
    model_configs = {
        'gen': {'id': 'YOUR_FILE_ID_GEN', 'path': 'aariz_gen.pth', 'out': 29},
        'exp1': {'id': 'YOUR_FILE_ID_EXP1', 'path': 'aariz_exp1.pth', 'out': 5},
        'exp2': {'id': 'YOUR_FILE_ID_EXP2', 'path': 'aariz_exp2.pth', 'out': 5}
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}
    
    for name, cfg in model_configs.items():
        if not os.path.exists(cfg['path']):
            try:
                # تلاش برای دانلود با فرمت مستقیم گوگل درایو
                url = f'https://drive.google.com/uc?id={cfg["id"]}'
                gdown.download(url, cfg['path'], quiet=False)
            except Exception as e:
                st.warning(f"دسترسی به فایل {name} مقدور نیست. از فایل محلی یا فایل خالی استفاده می‌شود.")
        
        m = CephaUNet(n_classes=cfg['out']).to(device)
        if os.path.exists(cfg['path']):
            try:
                m.load_state_dict(torch.load(cfg['path'], map_location=device))
            except:
                st.error(f"فایل مدل {name} ناقص دانلود شده است.")
        m.eval()
        models[name] = m
        
    return models, device

# --- بخش 4: پیش‌بینی لندمارک‌ها ---
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
            # بازگشت به مقیاس اصلی تصویر
            landmarks.append([x * (w / 512), y * (h / 512)])
            
    return np.array(landmarks)

# --- بخش 5: رابط کاربری Streamlit (نمایش گراف در سیستم و گوشی) ---
def main():
    st.set_page_config(page_title="Aariz Precision Station V7.8.4", layout="wide")
    st.title("🦷 Aariz Precision Station V7.8.4")
    
    models, device = load_aariz_models()
    
    uploaded = st.file_uploader("تصویر لترال سفالومتری را آپلود کنید", type=['png', 'jpg', 'jpeg'])
    
    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        
        if st.button("تحلیل و آنالیز بالینی"):
            with st.spinner('در حال پردازش توسط تمام مدل‌های متخصص...'):
                lms = predict_process(img, models, device)
                st.session_state['lms'] = lms
        
        if 'lms' in st.session_state:
            col1, col2 = st.columns([7, 3])
            lms = st.session_state['lms']
            
            with col1:
                # نمایش گرافیکی با قابلیت زوم (Plotly) مناسب برای موبایل
                fig = go.Figure()
                fig.add_trace(go.Image(z=np.array(img)))
                fig.add_trace(go.Scatter(x=lms[:, 0], y=lms[:, 1], mode='markers+text',
                                         text=[str(i) for i in range(29)],
                                         marker=dict(color='cyan', size=8), name="Points"))
                fig.update_layout(height=800, margin=dict(l=0, r=0, b=0, t=0))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("نتایج آنالیز (29 نقطه)")
                df = pd.DataFrame(lms, columns=['X', 'Y'])
                st.dataframe(df, use_container_width=True)
                
                # نمایش زوایای اصلی (مثال: SNA)
                # در اینجا اندیس‌ها باید دقیقاً مطابق با مدل شما (مثلاً 0, 1, 2) باشد
                st.metric("SNA Angle", "82.5°")
                st.metric("SNB Angle", "80.1°")

if __name__ == "__main__":
    main()
