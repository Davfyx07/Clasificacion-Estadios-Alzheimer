import os
import shutil
import random
import numpy as np
from PIL import Image

# ── CONFIGURACIÓN SENIOR ────────────────────────────────────
SEED          = 42
N_TRAIN_REF   = 5000  
IMG_SIZE      = 224   
DATA_DIR      = "/home/davfy/Escritorio/Alzheimer (Preprocessed Data)/" 
OUTPUT_DIR    = "/home/davfy/Escritorio/Vision/dataset_balanceado2"

CLASES = ["Non_Demented", "Very_Mild_Demented", "Mild_Demented", "Moderate_Demented"]

random.seed(SEED)
np.random.seed(SEED)

def preparar_dataset_senior():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    for split in ["train", "val", "test"]:
        for clase in CLASES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, clase), exist_ok=True)

    for clase in CLASES:
        ruta_clase = os.path.join(DATA_DIR, clase)
        fotos = [f for f in os.listdir(ruta_clase) if f.endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(fotos)

        # Split 80/10/10
        n = len(fotos)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        train_reales = fotos[:train_end]
        val_reales   = fotos[train_end:val_end]
        test_reales  = fotos[val_end:]

        # Copiar Reales (Sin tocar nada)
        for f in val_reales: shutil.copy(os.path.join(ruta_clase, f), os.path.join(OUTPUT_DIR, "val", clase, f))
        for f in test_reales: shutil.copy(os.path.join(ruta_clase, f), os.path.join(OUTPUT_DIR, "test", clase, f))
        for f in train_reales: shutil.copy(os.path.join(ruta_clase, f), os.path.join(OUTPUT_DIR, "train", clase, f))

        # Aumento OFFLINE solo para Train
        n_actual = len(train_reales)
        faltantes = N_TRAIN_REF - n_actual
        
        if faltantes > 0:
            print(f"📊 Clase {clase}: {n_actual} reales -> Generando {faltantes} sintéticas.")
            # LIMITACIÓN DE RIESGO: Si la clase es muy pequeña, el aumento es sutil
            for i in range(faltantes):
                img_path = os.path.join(ruta_clase, random.choice(train_reales))
                img = Image.open(img_path).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                
                angle = random.uniform(-7, 7) # Solo 7 grados
                zoom = random.uniform(0.95, 1.05) # Solo 5% de zoom
                
                img = img.rotate(angle)
                w, h = img.size
                img = img.crop((w*(1-1/zoom)/2, h*(1-1/zoom)/2, w*(1+1/zoom)/2, h*(1+1/zoom)/2))
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                
                img.save(os.path.join(OUTPUT_DIR, "train", clase, f"aug_{i}_{clase}.jpg"))

preparar_dataset_senior()