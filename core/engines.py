import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np
import cv2
import streamlit as st
from facenet_pytorch import MTCNN # <--- Library Baru (Ringan & Kompatibel)

# --- ARSITEKTUR MODEL RESNET (TETAP SAMA) ---
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

# --- ENGINE UTAMA ---
class FaceEngine:
    def __init__(self, model_path='models/model-absensi.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Model ArcFace (PyTorch)
        self.model = _IndonesianFaceModel(num_classes=68)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Model Load Error: {e}")

        # 2. Preprocessing Image
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 3. SETUP DETEKTOR (MTCNN versi PyTorch)
        # keep_all=True artinya deteksi semua wajah, bukan cuma 1
        self.detector = MTCNN(keep_all=True, device=self.device, min_face_size=40)

    def extract_face_coords(self, image_cv2):
        """
        Deteksi wajah menggunakan Facenet-PyTorch MTCNN.
        Output: (x, y, w, h)
        """
        if image_cv2 is None: return None
        
        height, width, _ = image_cv2.shape
        
        # MTCNN butuh RGB (PIL Image)
        img_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Deteksi (Mengembalikan koordinat kotak dan probabilitas)
        boxes, probs = self.detector.detect(img_pil)
        
        if boxes is None or len(boxes) == 0:
            return None
        
        # Jika terdeteksi banyak wajah, ambil yang probabilitasnya paling tinggi
        # Atau ambil yang ukurannya paling besar
        best_box_idx = np.argmax(probs)
        box = boxes[best_box_idx]
        
        # Format box dari facenet-pytorch adalah [x1, y1, x2, y2]
        x1, y1, x2, y2 = [int(b) for b in box]
        
        # Ubah ke format [x, y, w, h]
        x = max(0, x1)
        y = max(0, y1)
        w = min(width - x, x2 - x1)
        h = min(height - y, y2 - y1)
        
        # Validasi ukuran
        if w < 20 or h < 20:
            return None
            
        return (x, y, w, h)

    def get_embedding(self, face_crop):
        """Ubah wajah crop jadi vektor"""
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