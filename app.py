# Aariz Precision Station V7.8.16 (Updated Font Path: Vazir.ttf)
import os
import io
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------------------------------------------------
# Page Configuration & UI Settings
# ---------------------------------------------------------
st.set_page_config(
    page_title="Aariz Precision Station V7.8.16",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Aariz Precision Station V7.8.16 - Cephalometric Analysis System")

# ---------------------------------------------------------
# Font Configuration (Updated for Vazir.ttf)
# ---------------------------------------------------------
FONT_PATH = "Vazir.ttf"  # Exact match with repository file name

def register_pdf_fonts():
    """Registers the Vazir font for PDF generation if available."""
    if os.path.exists(FONT_PATH):
        try:
            pdfmetrics.registerFont(TTFont('Vazir', FONT_PATH))
            return True
        except Exception as e:
            st.warning(f"Error registering font '{FONT_PATH}': {e}")
            return False
    else:
        st.info(f"Font file '{FONT_PATH}' not found in root repository. Defaulting to system fonts.")
        return False

# Register font on app initialization
HAS_VAZIR_FONT = register_pdf_fonts()

# ---------------------------------------------------------
# Device Selection & Model Architecture Definitions
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class CephaUNet(nn.Module):
    """UNet Architecture for Cephalometric Landmark Heatmap Estimation"""
    def __init__(self, in_channels=1, out_channels=29):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up4(x)
        
        logits = self.outc(x)
        return logits

# ---------------------------------------------------------
# Landmark Definitions & Multi-Model Mapping
# ---------------------------------------------------------
LANDMARK_NAMES = [
    "Sella (S)", "Nasion (N)", "Orbitale (Or)", "Porion (Po)", "Subspinale (A)",
    "Supramentale (B)", "Pogonion (Pog)", "Menton (Me)", "Gnathion (Gn)", "Gonion (Go)",
    "Upper Incisor Tip (U1T)", "Upper Incisor Apex (U1A)", "Lower Incisor Tip (L1T)", "Lower Incisor Apex (L1A)",
    "Upper Molar Occlusal (U6M)", "Lower Molar Occlusal (L6M)", "Anterior Nasal Spine (ANS)", "Posterior Nasal Spine (PNS)",
    "Articulare (Ar)", "Basion (Ba)", "Condylon (Cd)", "Pterygoid (Pt)", "Basion-Nasion Point",
    "Soft Tissue Nasion", "Soft Tissue Pronasale", "Soft Tissue Labrale Superius", "Soft Tissue Labrale Inferius",
    "Soft Tissue Pogonion", "Soft Tissue Menton"
]

# Map specific specialized regions to Specialist / TMJ Specialist models
SPECIALIST_LANDMARKS = [4, 5, 10, 11, 12, 13]  # Dentofacial / Subspinale-Incise regions
TMJ_LANDMARKS = [18, 19, 20]                   # TMJ / Condyle / Articulare regions

MODEL_PATHS = {
    "general": "checkpoint_unet_clinical.pth",
    "specialist": "specialist_pure_model.pth",
    "tmj": "tmj_specialist_model.pth"
}

# ---------------------------------------------------------
# Dynamic Model Loader
# ---------------------------------------------------------
@st.cache_resource
def load_all_models():
    """Loads general and specialist models into memory."""
    models = {}
    for key, path in MODEL_PATHS.items():
        model = CephaUNet(in_channels=1, out_channels=29)
        if os.path.exists(path):
            try:
                state_dict = torch.load(path, map_location=device)
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                models[key] = model
            except Exception as e:
                st.error(f"Failed loading model weights for {key} from {path}: {e}")
                models[key] = None
        else:
            st.warning(f"Model file {path} not found. Running under fallback/partial inference mode.")
            models[key] = None
    return models

models = load_all_models()

# ---------------------------------------------------------
# Image Processing & Prediction Pipeline
# ---------------------------------------------------------
def preprocess_image(image: Image.Image, target_size=(512, 512)):
    """Preprocesses input image for CephaUNet model."""
    img_gray = image.convert("L")
    img_resized = img_gray.resize(target_size)
    tensor = T.ToTensor()(img_resized)
    tensor = T.Normalize(mean=[0.5], std=[0.5])(tensor)
    return tensor.unsqueeze(0).to(device), img_gray.size

def extract_landmarks_from_heatmaps(heatmaps, original_size, target_size=(512, 512)):
    """Converts heatmaps into (X, Y) pixel coordinates scaled to original image."""
    heatmaps_np = heatmaps.squeeze(0).cpu().detach().numpy()
    landmarks = []
    orig_w, orig_h = original_size
    scale_x = orig_w / target_size[0]
    scale_y = orig_h / target_size[1]
    
    for idx in range(heatmaps_np.shape[0]):
        hm = heatmaps_np[idx]
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        coord_x = float(x * scale_x)
        coord_y = float(y * scale_y)
        landmarks.append((coord_x, coord_y))
    return landmarks

def ensemble_predict(models, tensor_img, orig_size):
    """Combines general, specialist, and TMJ models for optimal landmark precision."""
    preds = {}
    for key, model in models.items():
        if model is not None:
            with torch.no_grad():
                out = model(tensor_img)
                preds[key] = extract_landmarks_from_heatmaps(out, orig_size)
    
    # Merge strategy
    final_landmarks = []
    if "general" in preds:
        final_landmarks = list(preds["general"])
    else:
        final_landmarks = [(0.0, 0.0)] * 29

    if "specialist" in preds:
        for idx in SPECIALIST_LANDMARKS:
            final_landmarks[idx] = preds["specialist"][idx]

    if "tmj" in preds:
        for idx in TMJ_LANDMARKS:
            final_landmarks[idx] = preds["tmj"][idx]

    return final_landmarks

# ---------------------------------------------------------
# Cephalometric Analysis Computations
# ---------------------------------------------------------
def calculate_angle(p1, p2, p3):
    """Calculates angle (in degrees) formed at vertex p2 by points p1-p2-p3."""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(angle))

def run_cephalometric_analysis(landmarks):
    """Computes clinical cephalometric parameters from predicted landmarks."""
    analysis = {}
    
    # SNA (Sella-Nasion-Subspinale)
    if len(landmarks) > 4:
        analysis["SNA"] = calculate_angle(landmarks[0], landmarks[1], landmarks[4])
    
    # SNB (Sella-Nasion-Supramentale)
    if len(landmarks) > 5:
        analysis["SNB"] = calculate_angle(landmarks[0], landmarks[1], landmarks[5])
    
    # ANB (SNA - SNB)
    if "SNA" in analysis and "SNB" in analysis:
        analysis["ANB"] = analysis["SNA"] - analysis["SNB"]

    # FMA (Porion-Orbitale to Gonion-Menton)
    if len(landmarks) > 9:
        p_po, p_or = landmarks[3], landmarks[2]
        p_go, p_me = landmarks[9], landmarks[7]
        v_fh = np.array([p_or[0] - p_po[0], p_or[1] - p_po[1]])
        v_mp = np.array([p_me[0] - p_go[0], p_me[1] - p_go[1]])
        n1, n2 = np.linalg.norm(v_fh), np.linalg.norm(v_mp)
        if n1 > 0 and n2 > 0:
            cos_a = np.dot(v_fh, v_mp) / (n1 * n2)
            analysis["FMA"] = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    return analysis

# ---------------------------------------------------------
# PDF Report Generation Component
# ---------------------------------------------------------
def generate_pdf_report(image: Image.Image, landmarks, analysis_results):
    """Generates a professional clinical PDF report."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    story = []
    
    font_name = 'Vazir' if HAS_VAZIR_FONT else 'Helvetica'
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontName=font_name, fontSize=18, leading=22, alignment=1)
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontName=font_name, fontSize=10, leading=14)
    
    story.append(Paragraph("Aariz Cephalometric Analysis Report", title_style))
    story.append(Spacer(1, 15))
    
    # Add Overlay Landmark Image
    img_draw = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img_draw)
    for idx, (x, y) in enumerate(landmarks):
        draw.ellipse([x-4, y-4, x+4, y+4], fill="red", outline="yellow")
    
    img_buffer = io.BytesIO()
    img_draw.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    
    story.append(RLImage(img_buffer, width=240, height=240))
    story.append(Spacer(1, 15))
    
    # Analysis Table
    table_data = [["Parameter", "Measured Value", "Reference Range"]]
    ref_ranges = {"SNA": "82.0° ± 2.0°", "SNB": "80.0° ± 2.0°", "ANB": "2.0° ± 1.0°", "FMA": "25.0° ± 3.0°"}
    
    for k, v in analysis_results.items():
        ref = ref_ranges.get(k, "N/A")
        table_data.append([k, f"{v:.2f}°", ref])
        
    t = Table(table_data, colWidths=[150, 150, 150])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E88E5")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,-1), font_name),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    story.append(t)
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ---------------------------------------------------------
# Sidebar & File Upload Interface
# ---------------------------------------------------------
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Cephalogram (PNG / JPG / BMP)", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Cephalogram Image", use_column_width=True)
    
    if st.button("Run Precision Analysis"):
        with st.spinner("Executing CephaUNet Specialist Ensembles..."):
            tensor_img, orig_size = preprocess_image(image)
            landmarks = ensemble_predict(models, tensor_img, orig_size)
            analysis_results = run_cephalometric_analysis(landmarks)
            
            st.success("Analysis Complete!")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Landmark Coordinates")
                df_landmarks = pd.DataFrame(landmarks, columns=["X (px)", "Y (px)"])
                df_landmarks.index = LANDMARK_NAMES[:len(landmarks)]
                st.dataframe(df_landmarks)
                
            with col2:
                st.subheader("Cephalometric Parameters")
                df_analysis = pd.DataFrame(list(analysis_results.items()), columns=["Parameter", "Value (deg)"])
                st.dataframe(df_analysis)
                
                # PDF Generation Action
                pdf_bytes = generate_pdf_report(image, landmarks, analysis_results)
                st.download_button(
                    label="Download Clinical PDF Report",
                    data=pdf_bytes,
                    file_name="Aariz_Cephalometric_Report.pdf",
                    mime="application/pdf"
                )
else:
    st.info("Please upload a lateral cephalogram image from the sidebar to begin processing.")
