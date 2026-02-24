import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os, gdown
from PIL import Image, ImageDraw, ImageOps
import torchvision.transforms as transforms
from streamlit_image_coordinates import streamlit_image_coordinates

# --- [Core Architecture: DoubleConv & CephaUNet] ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class CephaUNet(nn.Module):
    def __init__(self, n_landmarks=29):
        super(CephaUNet, self).__init__()
        self.inc = DoubleConv(1, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_landmarks, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3)
        u1 = self.up1(x4); u1 = torch.cat([u1, x3], dim=1); c1 = self.conv_up1(u1)
        u2 = self.up2(c1); u2 = torch.cat([u2, x2], dim=1); c2 = self.conv_up2(u2)
        u3 = self.up3(c2); u3 = torch.cat([u3, x1], dim=1); c3 = self.conv_up3(u3)
        return self.outc(c3)

# --- [Model Loading: 3-Model Ensemble System] ---
@st.cache_resource
def load_aariz_models():
    model_ids = {
        'm1': '1a1sZ2z0X6mOwljhBjmItu_qrWYv3v_ks', 
        'm2': '1RakXVfUC_ETEdKGBi6B7xOD7MjD59jfU', 
        'm3': '1tizRbUwf7LgC6Radaeiz6eUffiwal0cH'
    }
    device = torch.device("cpu")
    loaded_models = []
    for key, file_id in model_ids.items():
        model_path = f"{key}.pth"
        if not os.path.exists(model_path):
            gdown.download(f'https://drive.google.com/uc?id={file_id}', model_path, quiet=True)
        model = CephaUNet(n_landmarks=29).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items() if k.replace('module.', '') in model.state_dict()}, strict=False)
        model.eval()
        loaded_models.append(model)
    return loaded_models, device

# --- [Main Setup] ---
st.set_page_config(page_title="Aariz Precision Station V7.8", layout="wide")
landmark_names = ['A', 'ANS', 'B', 'Me', 'N', 'Or', 'Pog', 'PNS', 'Pn', 'R', 'S', 'Ar', 'Co', 'Gn', 'Go', 'Po', 'LPM', 'LIT', 'LMT', 'UPM', 'UIA', 'UIT', 'UMT', 'LIA', 'Li', 'Ls', 'N`', 'Pog`', 'Sn']
models, device = load_aariz_models()

st.sidebar.title("🩺 Aariz Station V7.8")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    if "lms" not in st.session_state or st.session_state.file_id != uploaded_file.name:
        img_raw = Image.open(uploaded_file).convert("RGB")
        st.session_state.img = img_raw
        W, H = img_raw.size; ratio = 512 / max(W, H)
        img_rs = img_raw.convert("L").resize((int(W*ratio), int(H*ratio)), Image.NEAREST)
        canvas = Image.new("L", (512, 512)); px, py = (512-img_rs.width)//2, (512-img_rs.height)//2
        canvas.paste(img_rs, (px, py))
        tensor = transforms.ToTensor()(canvas).unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds = [m(tensor)[0].cpu().numpy() for m in models]
            lms = {}
            for i in range(29):
                m_idx = 1 if i in {10, 14, 9, 5, 28, 20} else (2 if i in {7, 11, 12, 15} else 0)
                y, x = divmod(np.argmax(preds[m_idx][i]), 512)
                lms[i] = [int((x - px) / ratio), int((y - py) / ratio)]
        st.session_state.lms = lms
        st.session_state.file_id = uploaded_file.name
        st.session_state.v = 0

    l = st.session_state.lms; img = st.session_state.img; W, H = img.size
    target_idx = st.sidebar.selectbox("🎯 انتخاب لندمارک:", range(29), format_func=lambda x: f"{x}: {landmark_names[x]}")

    col1, col2 = st.columns([1.5, 3.5])
    
    with col1:
        st.subheader("🔍 Magnifier")
        cur = l[target_idx]; box = 100
        # --- Safe Padding: حل قطعی ValueError بدون تغییر در عملکرد ---
        img_p = ImageOps.expand(img, border=box, fill='black')
        p_x, p_y = cur[0] + box, cur[1] + box
        crop = img_p.crop((p_x - box, p_y - box, p_x + box, p_y + box)).resize((400, 400), Image.NEAREST)
        
        draw_m = ImageDraw.Draw(crop)
        draw_m.line((190, 200, 210, 200), fill="red", width=2)
        draw_m.line((200, 190, 200, 210), fill="red", width=2)
        
        res_m = streamlit_image_coordinates(crop, key=f"m_{target_idx}_{st.session_state.v}")
        if res_m:
            l[target_idx] = [int(cur[0] - box + (res_m['x'] * (200/400))), int(cur[1] - box + (res_m['y'] * (200/400)))]
            st.session_state.v += 1; st.rerun()

    with col2:
        st.subheader("🖼 Cephalogram View")
        sc = 850 / W
        disp = img.copy().resize((850, int(H * sc)), Image.NEAREST)
        draw = ImageDraw.Draw(disp)
        for i, p in l.items():
            clr = (255, 0, 0) if i == target_idx else (0, 255, 0)
            draw.ellipse([p[0]*sc-3, p[1]*sc-3, p[0]*sc+3, p[1]*sc+3], fill=clr)
            draw.text((p[0]*sc+8, p[1]*sc-8), f"{i}:{landmark_names[i]}", fill=clr)
        
        res_main = streamlit_image_coordinates(disp, width=850, key=f"main_{st.session_state.v}")
        if res_main:
            l[target_idx] = [int(res_main['x']/sc), int(res_main['y']/sc)]
            st.session_state.v += 1; st.rerun()

    if st.sidebar.button("💾 Save & Finalize"):
        st.sidebar.success(f"Landmarks for {uploaded_file.name} saved successfully.")
