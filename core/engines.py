# core/engines.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np
import cv2
import streamlit as st
from facenet_pytorch import MTCNN
from torch.quantization import quantize_dynamic

class _IndonesianFaceModel(nn.Module):
    def __init__(self, num_classes=68):
        super(_IndonesianFaceModel, self).__init__()
        backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        input_dim = 2048
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(0.4)
        self.fc_embedding = nn.Linear(input_dim, 512)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bn_input(x)
        x = self.dropout(x)
        return self.fc_embedding(x)

class FaceEngine:
    def __init__(self, model_path='models/model-absensi.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = _IndonesianFaceModel(num_classes=68)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
            self.model.to(self.device)
            self.model.eval()

            # --- OPTIMASI MEMORI: QUANTIZATION ---
            if self.device.type == 'cpu':
                self.model = quantize_dynamic(
                    self.model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                )

        except Exception as e:
            print(f"Model Load Error: {e}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.detector = MTCNN(keep_all=True, device=self.device, min_face_size=40, thresholds=[0.6, 0.7, 0.7])

    def extract_face_coords(self, image_cv2):
        if image_cv2 is None: return None
        height, width, _ = image_cv2.shape
        img_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        try:
            boxes, probs = self.detector.detect(img_pil)
            if boxes is None or len(boxes) == 0: return None
            
            best_idx = np.argmax(probs)
            box = boxes[best_idx]
            x1, y1, x2, y2 = [int(b) for b in box]
            
            # Padding 0 (Ngepas Wajah) biar background tidak mengganggu analisis tekstur
            x = max(0, x1)
            y = max(0, y1)
            w = min(width - x, x2 - x1)
            h = min(height - y, y2 - y1)
            
            if w < 20 or h < 20: return None
            return (x, y, w, h)
        except Exception:
            return None

    def get_embedding(self, face_crop):
        if face_crop is None or face_crop.size == 0: return np.zeros(512)
        img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            embedding = self.model(img_tensor)
        return embedding.cpu().numpy()[0]
    
    def calculate_average_embedding(self, embeddings_list):
        if not embeddings_list: return None
        stack = np.stack(embeddings_list)
        mean_emb = np.mean(stack, axis=0)
        return mean_emb / np.linalg.norm(mean_emb)

    # --- ANTI SPOOFING (TEXTURE ANALYSIS) ---
    def check_liveness(self, face_crop):
        """
        Menggabungkan 2 metode:
        1. Laplacian Variance (Ketajaman Tekstur)
        2. Fourier Transform (Deteksi Pola Layar/Digital)
        """
        if face_crop is None or face_crop.size == 0:
            return False, 0.0
            
        # 1. Analisis Tekstur (Laplacian)
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Analisis Frekuensi (Fourier Transform)
        # Mendeteksi Moiré Pattern (Garis-garis halus layar HP)
        # Jika foto asli, frekuensi tinggi biasanya acak (random).
        # Jika foto layar, ada pola frekuensi tertentu.
        
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-5) # +1e-5 biar ga log(0)
        
        # Hitung rata-rata magnitude frekuensi tinggi
        # (Indikator sederhana: Wajah asli punya spektrum lebih "lembut")
        fourier_score = np.mean(magnitude_spectrum)
        
        # --- FORMULA GABUNGAN (RAHASIA DAPUR) ---
        # Kita bobotkan skornya.
        # Laplacian tinggi = Bagus (Tajam)
        # Fourier tinggi = Buruk (Banyak noise digital)
        
        # Trik: Kita jadikan satu skor final
        # Wajah asli biasanya Laplacian > 50 dan Fourier < 150
        
        # Kita pakai Laplacian sebagai skor utama karena paling stabil untuk demo
        # Tapi kita kurangi nilainya jika Fouriernya mencurigakan
        
        final_score = laplacian_score
        
        # Jika terdeteksi pola digital kuat (Fourier tinggi), kita pangkas skornya
        if fourier_score > 160: 
            final_score = final_score * 0.5 # Penalty 50%
            
        is_real = final_score > 60.0 # Ambang batas default (bisa diatur admin)
        
        return is_real, final_score