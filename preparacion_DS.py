"""
preparacion_DS.py - Versión Optimizada para Dataset Mendeley/Kaggle
- Redimensiona a 224x224 (estándar para modelos de IA).
- Balanceo exacto a 8,750 imágenes por clase en TRAIN (Total 35k).
- VAL y TEST: 100% imágenes reales sin alteraciones.
- Augmentation médico: Solo rotación y zoom leves (Sin Flips).
"""

import os
import shutil
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ── CONFIGURACIÓN DE RUTAS ──────────────────────────────────
SEED          = 42
N_TRAIN_REF   = 8750  # Objetivo por clase en Train
IMG_SIZE      = 224   # Tamaño para EfficientNet/ResNet
DATA_DIR      = "/home/davfy/Escritorio/Alzheimer (Preprocessed Data)/" 
OUTPUT_DIR    = "/home/davfy/Escritorio/Vision/dataset_balanceado"

# Nombres exactos de tus carpetas según la imagen que subiste
CLASES = [
    "Non_Demented",
    "Very_Mild_Demented",
    "Mild_Demented",
    "Moderate_Demented"
]

random.seed(SEED)
np.random.seed(SEED)

# ── FUNCIONES DE APOYO ──────────────────────────────────────

def limpiar_y_preparar():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    for split in ["train", "val", "test"]:
        for clase in CLASES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, clase), exist_ok=True)

def aumentar_medico(img_pil):
    """Aplica transformaciones leves que no alteran la anatomía médica."""
    # 1. Rotación leve (máximo 8 grados)
    angulo = random.uniform(-8, 8)
    img = img_pil.rotate(angulo, resample=Image.BICUBIC)
    
    # 2. Zoom muy ligero (entre 0.95 y 1.05)
    zoom = random.uniform(0.95, 1.05)
    w, h = img.size
    new_w, new_h = int(w * zoom), int(h * zoom)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    
    # Recortar o rellenar para mantener exactamente 224x224
    left = (new_w - IMG_SIZE) / 2
    top = (new_h - IMG_SIZE) / 2
    img = img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))
    
    # Asegurar que el tamaño final sea correcto
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        
    return img

# ── PROCESO PRINCIPAL ───────────────────────────────────────

print("🚀 Iniciando preparación del dataset balanceado...")
limpiar_y_preparar()

conteo_original = {}
conteo_final = {"train": [], "val": [], "test": []}

for clase in CLASES:
    ruta_clase = os.path.join(DATA_DIR, clase)
    imagenes = [os.path.join(ruta_clase, f) for f in os.listdir(ruta_clase) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(imagenes)
    
    conteo_original[clase] = len(imagenes)
    
    # 1. Split 80% Train, 10% Val, 10% Test (sobre las reales)
    n = len(imagenes)
    train_reales = imagenes[:int(n * 0.8)]
    val_reales   = imagenes[int(n * 0.8):int(n * 0.9)]
    test_reales  = imagenes[int(n * 0.9):]
    
    print(f"📦 Procesando {clase}: {n} imágenes reales encontradas.")

    # 2. Copiar Reales a sus destinos
    splits = {"train": train_reales, "val": val_reales, "test": test_reales}
    for split_name, lista_imgs in splits.items():
        for i, img_path in enumerate(lista_imgs):
            img = Image.open(img_path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            nombre_base = f"{clase}_{i:05d}_real.jpg"
            img.save(os.path.join(OUTPUT_DIR, split_name, clase, nombre_base))

    # 3. AUMENTO SOLO EN TRAIN para llegar a N_TRAIN_REF (8750)
    imgs_en_train = len(train_reales)
    faltantes = N_TRAIN_REF - imgs_en_train
    
    if faltantes > 0:
        print(f"  ✨ Generando {faltantes} imágenes sintéticas para {clase}...")
        for i in range(faltantes):
            # Elegir una imagen real de train al azar para transformarla
            img_original_path = random.choice(train_reales)
            img_pil = Image.open(img_original_path).convert("RGB")
            img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            
            img_aug = aumentar_medico(img_pil)
            nombre_aug = f"{clase}_{i:05d}_aug.jpg"
            img_aug.save(os.path.join(OUTPUT_DIR, "train", clase, nombre_aug))

# ── GENERACIÓN DE GRÁFICA DE CONTROL ────────────────────────
print("\n📊 Generando reporte visual...")

labels = CLASES
orig_vals = [conteo_original[c] for c in CLASES]
final_train_vals = [len(os.listdir(os.path.join(OUTPUT_DIR, "train", c))) for c in CLASES]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, orig_vals, width, label='Original (Total Real)')
ax.bar(x + width/2, final_train_vals, width, label='Final (Train Balanceado)')

ax.set_ylabel('Número de Imágenes')
ax.set_title('Distribución de Clases: Antes vs Después del Balanceo')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15)
ax.legend()

plt.tight_layout()
# Guardar gráfica en la carpeta de visión para tu informe
grafica_path = "/home/davfy/Escritorio/Vision/distribucion_dataset.png"
plt.savefig(grafica_path)
print(f"✅ Proceso terminado. Gráfica guardada en: {grafica_path}")