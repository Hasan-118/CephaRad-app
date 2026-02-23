import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import gdown
import os
import plotly.graph_objects as go

# --- CONFIGURATION & GOLD STANDARD REFERENCE ---
VERSION = "V7.8"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# مدل عمومی و متخصص‌ها (طبق دستور شما: ۳ مدل بارگذاری شوند)
MODELS_INFO = {
    "General": {"id": "YOUR_GD_ID_1", "path": f"{MODEL_DIR}/general_v78.pth"},
    "Expert_1": {"id": "YOUR_GD_ID_2", "path": f"{MODEL_DIR}/expert1_v78.pth"},
    "Expert_2": {"id": "YOUR_GD_ID_3", "path": f"{MODEL_DIR}/expert2_v78.pth"}
}

# --- ARCHITECTURE (DoubleConv & CephaUNet) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class CephaUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=29):
        super(CephaUNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling
        features = [64, 128, 256, 512]
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Upsampling
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip
