"""
preparacion_DS.py
Prepara el dataset OASIS balanceado con augmentation SOLO en train.
- Train: 8750 imágenes por clase (reales + sintéticas)
- Val/Test: solo imágenes reales, sin augmentation
Output: /home/davfy/Escritorio/Vision/dataset_balanceado/
"""

import os
import shutil
import random
import numpy as np
from PIL import Image, ImageEnhance
import cv2

# ── Configuración ──────────────────────────────────────────
SEED          = 42
N_REF         = 8750
IMG_SIZE      = 224
DATA_DIR      = "/home/davfy/Documentos/Data"
OUTPUT_DIR    = "/home/davfy/Escritorio/Vision/dataset_balanceado"

CLASES = [
    "Non Demented",
    "Very mild Dementia",
    "Mild Dementia",
    "Moderate Dementia"
]

# Caso especial Moderate — solo 2 pacientes en OASIS-1
MODERATE_TRAIN    = "OAS1_0308"
MODERATE_VAL_TEST = "OAS1_0351"

random.seed(SEED)
np.random.seed(SEED)

# ── Augmentation (SOLO para train) ────────────────────────
def augmentar(img_pil):
    img = np.array(img_pil.convert("L"))

    # Rotación
    if random.random() < 0.5:
        angle = random.uniform(-10, 10)
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))

    # Escalado
    if random.random() < 0.4:
        s = random.uniform(0.95, 1.05)
        h, w = img.shape
        img = cv2.resize(img, (int(w*s), int(h*s)))
        img = cv2.resize(img, (w, h))

    # Brillo
    if random.random() < 0.5:
        beta = random.uniform(0.90, 1.10)
        img = np.clip(img * beta, 0, 255).astype(np.uint8)

    # Contraste
    if random.random() < 0.5:
        alpha = random.uniform(0.90, 1.10)
        mean  = img.mean()
        img   = np.clip(mean + alpha * (img - mean), 0, 255).astype(np.uint8)

    return Image.fromarray(img).convert("RGB")

# ── Obtener patient_id del nombre de archivo ──────────────
def get_patient_id(fname):
    partes = fname.split("_")
    # OAS1_0308_MR1_... → OAS1_0308
    if len(partes) >= 2:
        return f"{partes[0]}_{partes[1]}"
    return fname

# ── Copiar + resize (sin augmentation) ───────────────────
def copiar(src, dst):
    img = Image.open(src).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    img.save(dst, quality=95)

# ── Main ──────────────────────────────────────────────────
def preparar():
    # Limpiar output
    if os.path.exists(OUTPUT_DIR):
        print("Limpiando carpeta de salida antigua...")
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val", "test"]:
        for clase in CLASES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, clase), exist_ok=True)

    print(f"Output: {OUTPUT_DIR}\n")

    for clase in CLASES:
        ruta_clase = os.path.join(DATA_DIR, clase)
        if not os.path.exists(ruta_clase):
            print(f"ERROR: No se encuentra {ruta_clase}")
            continue

        # Agrupar imágenes por paciente
        archivos = sorted([f for f in os.listdir(ruta_clase)
                           if f.lower().endswith((".jpg", ".png", ".jpeg"))])

        pacientes = {}
        for f in archivos:
            pid = get_patient_id(f)
            pacientes.setdefault(pid, []).append(f)

        lista_pids = sorted(pacientes.keys())

        # ── Partición por paciente ────────────────────────
        if clase == "Moderate Dementia":
            # Caso especial: solo 2 pacientes
            train_pids   = [p for p in lista_pids if MODERATE_TRAIN    in p]
            val_pids     = [p for p in lista_pids if MODERATE_VAL_TEST in p]
            test_pids    = val_pids  # mismo paciente, declarado como limitación
        else:
            random.shuffle(lista_pids)
            n       = len(lista_pids)
            n_train = int(n * 0.70)
            n_val   = int(n * 0.15)
            train_pids = lista_pids[:n_train]
            val_pids   = lista_pids[n_train:n_train + n_val]
            test_pids  = lista_pids[n_train + n_val:]

        # Recopilar rutas por split
        def rutas_de(pids):
            return [os.path.join(ruta_clase, f)
                    for pid in pids for f in pacientes[pid]]

        rutas_train = rutas_de(train_pids)
        rutas_val   = rutas_de(val_pids)
        rutas_test  = rutas_de(test_pids)
        rutas_orig  = rutas_train.copy()  # originales para augmentation

        # ── TRAIN: submuestreo + augmentation ────────────
        out_train = os.path.join(OUTPUT_DIR, "train", clase)

        # Submuestreo si hay más de N_REF
        if len(rutas_train) > N_REF:
            rutas_train = random.sample(rutas_train, N_REF)

        # Copiar originales
        for i, ruta in enumerate(rutas_train):
            dst = os.path.join(out_train, f"{clase.replace(' ','_')}_{i:05d}.jpg")
            copiar(ruta, dst)

        # Augmentation si faltan para llegar a N_REF
        if len(rutas_train) < N_REF:
            faltan   = N_REF - len(rutas_train)
            contador = len(rutas_train)
            idx      = 0
            print(f"  {clase}: {len(rutas_train)} reales → generando {faltan} sintéticas...")
            while faltan > 0:
                src = rutas_orig[idx % len(rutas_orig)]
                img_aug = augmentar(Image.open(src))
                img_aug = img_aug.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
                fname = f"{clase.replace(' ','_')}_aug_{contador:05d}.jpg"
                img_aug.save(os.path.join(out_train, fname), quality=95)
                faltan   -= 1
                contador += 1
                idx      += 1

        # ── VAL y TEST: solo reales, sin tocar ───────────
        for i, ruta in enumerate(rutas_val):
            dst = os.path.join(OUTPUT_DIR, "val", clase,
                               f"{clase.replace(' ','_')}_{i:05d}.jpg")
            copiar(ruta, dst)

        for i, ruta in enumerate(rutas_test):
            dst = os.path.join(OUTPUT_DIR, "test", clase,
                               f"{clase.replace(' ','_')}_{i:05d}.jpg")
            copiar(ruta, dst)

        n_tr = len(os.listdir(out_train))
        n_vl = len(rutas_val)
        n_ts = len(rutas_test)
        print(f"  ✅ {clase}: Train={n_tr} | Val={n_vl} | Test={n_ts}")

    # ── Resumen final ─────────────────────────────────────
    print("\n── Resumen final ──────────────────────────────")
    total = 0
    for split in ["train", "val", "test"]:
        subtotal = 0
        print(f"\n  {split.upper()}:")
        for clase in CLASES:
            p = os.path.join(OUTPUT_DIR, split, clase)
            n = len(os.listdir(p)) if os.path.exists(p) else 0
            print(f"    {clase}: {n}")
            subtotal += n
        print(f"    SUBTOTAL: {subtotal}")
        total += subtotal
    print(f"\n  TOTAL DATASET: {total} imágenes")
    print(f"\n✅ Dataset listo en: {OUTPUT_DIR}")
    print("NOTA: Train balanceado a 8750/clase con augmentation.")
    print("      Val y Test con imágenes reales únicamente.")
    print("      Moderate Dementia: val y test comparten OAS1_0351 (limitación declarada).")

if __name__ == "__main__":
    preparar()