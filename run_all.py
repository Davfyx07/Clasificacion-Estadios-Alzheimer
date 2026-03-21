"""
run_all.py
Script maestro — ejecuta todo en orden:
1. Preparación del dataset (si no existe)
2. EfficientNetV2S + CBAM
3. MobileNetV3Large + CBAM
4. ResNet50 + CBAM
5. Comparativa final

Ejecútalo y vete a dormir :)
"""

import os
import sys
import time
import subprocess

BASE = "/home/davfy/Escritorio/Vision"
SCRIPTS = BASE  # carpeta donde están los scripts

def log(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")

def correr(script):
    resultado = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS, script)],
        check=True
    )
    return resultado

t_total = time.time()

# ── 1. Preparar dataset ────────────────────────────────────
dataset_dir = os.path.join(BASE, "dataset_balanceado", "train")
if os.path.exists(dataset_dir):
    log("Dataset ya existe — saltando preparacion_DS.py")
else:
    log("PASO 1: Preparando dataset...")
    correr("preparacion_DS.py")

# ── 2. EfficientNetV2S ─────────────────────────────────────
log("PASO 2: Entrenando EfficientNetV2S + CBAM")
correr("train_efficientnet.py")

# ── 3. MobileNetV3 ─────────────────────────────────────────
log("PASO 3: Entrenando MobileNetV3Large + CBAM")
correr("train_mobilenet.py")

# ── 4. ResNet50 ────────────────────────────────────────────
log("PASO 4: Entrenando ResNet50 + CBAM")
correr("train_resnet.py")

# ── 5. Comparativa ─────────────────────────────────────────
log("PASO 5: Generando comparativa final")
correr("comparar_modelos.py")

mins = (time.time() - t_total) / 60
log(f"TODO LISTO en {mins:.1f} minutos — revisa /home/davfy/Escritorio/Vision/resultados/")