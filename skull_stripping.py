"""
skull_stripping.py
Elimina el cráneo, ojos y tejido externo de las imágenes de RM cerebral OASIS.
Método: Umbralización + morfología + contorno mayor (OpenCV)

Input:  /home/davfy/Escritorio/Data/  (imágenes originales OASIS)
Output: /home/davfy/Escritorio/Data_stripped/  (solo cerebro)

Genera además una muestra visual de 16 imágenes (antes/después)
para verificar la calidad antes de usar el dataset.
"""

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# ── Configuración ──────────────────────────────────────────
DATA_DIR    = "/home/davfy/Escritorio/Data"
OUTPUT_DIR  = "/home/davfy/Escritorio/Data_stripped"
MUESTRA_DIR = "/home/davfy/Escritorio/Vision"
IMG_SIZE    = 224  # Tamaño final

CLASES = [
    "Non Demented",
    "Very mild Dementia",
    "Mild Dementia",
    "Moderate Dementia",
]

# ── Skull stripping ────────────────────────────────────────
def skull_strip(img_bgr):
    """
    Elimina el cráneo y tejido externo de una imagen de RM cerebral.
    Retorna la imagen con fondo negro (solo cerebro visible).

    Pasos:
    1. Escala de grises
    2. Blur gaussiano para reducir ruido
    3. Umbral de Otsu para separar cerebro/fondo
    4. Operaciones morfológicas para limpiar la máscara
    5. Contorno más grande = cerebro
    6. Máscara convexa para rellenar huecos internos
    7. Aplicar máscara a imagen original
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Blur para reducir ruido antes de umbralizar
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Umbral de Otsu — automático, se adapta a cada imagen
    _, thresh = cv2.threshold(blur, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Operaciones morfológicas para limpiar artefactos pequeños
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=2)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Si no encuentra nada, devolver la imagen original
        return img_bgr

    # El contorno más grande es el cerebro
    largest = max(contours, key=cv2.contourArea)

    # Máscara convexa para rellenar huecos internos del cerebro
    hull    = cv2.convexHull(largest)
    mask    = np.zeros_like(gray)
    cv2.fillPoly(mask, [hull], 255)

    # Dilatar la máscara ligeramente para no cortar bordes del cerebro
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask    = cv2.dilate(mask, kernel2, iterations=2)

    # Aplicar máscara — fondo queda en negro
    result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

    return result

def procesar_imagen(src_path, dst_path):
    """Lee, aplica skull strip, redimensiona y guarda."""
    img = cv2.imread(src_path)
    if img is None:
        return False

    # Skull stripping
    stripped = skull_strip(img)

    # Redimensionar a tamaño estándar
    stripped = cv2.resize(stripped, (IMG_SIZE, IMG_SIZE),
                           interpolation=cv2.INTER_LANCZOS4)

    cv2.imwrite(dst_path, stripped)
    return True

# ── Generar muestra visual ─────────────────────────────────
def generar_muestra(muestras):
    """
    Genera una imagen con 4 columnas × 4 filas mostrando antes/después.
    muestras: lista de (ruta_original, ruta_procesada, clase)
    """
    n = min(8, len(muestras))  # máximo 8 pares
    fig = plt.figure(figsize=(16, n * 2.5))
    fig.suptitle("Skull Stripping — Antes vs Después\n(OASIS dataset)",
                  fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(n, 2, hspace=0.4, wspace=0.1)

    for i, (orig_path, strip_path, clase) in enumerate(muestras[:n]):
        orig  = cv2.cvtColor(cv2.imread(orig_path),  cv2.COLOR_BGR2GRAY)
        strip = cv2.cvtColor(cv2.imread(strip_path), cv2.COLOR_BGR2GRAY)

        ax1 = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1])

        ax1.imshow(orig,  cmap="gray")
        ax1.set_title(f"Original — {clase}", fontsize=9)
        ax1.axis("off")

        ax2.imshow(strip, cmap="gray")
        ax2.set_title("Skull stripped", fontsize=9)
        ax2.axis("off")

    path = os.path.join(MUESTRA_DIR, "skull_strip_muestra.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nMuestra visual guardada en: {path}")
    print("¡Revísala antes de usar el dataset procesado!")

# ── Main ──────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  SKULL STRIPPING — Dataset OASIS")
    print("=" * 60)
    print(f"  Origen:  {DATA_DIR}")
    print(f"  Destino: {OUTPUT_DIR}")
    print(f"  Tamaño:  {IMG_SIZE}×{IMG_SIZE} px")
    print("=" * 60)

    # Crear carpetas de destino
    for clase in CLASES:
        os.makedirs(os.path.join(OUTPUT_DIR, clase), exist_ok=True)

    total_ok   = 0
    total_fail = 0
    muestras   = []  # para la gráfica

    for clase in CLASES:
        src_dir = os.path.join(DATA_DIR, clase)
        dst_dir = os.path.join(OUTPUT_DIR, clase)

        if not os.path.exists(src_dir):
            print(f"\n⚠️  No encontrada: {src_dir}")
            continue

        archivos = sorted([
            f for f in os.listdir(src_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        print(f"\n── {clase}: {len(archivos)} imágenes")

        clase_ok = 0
        for fname in tqdm(archivos, desc=f"  {clase[:15]:<15}"):
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)

            ok = procesar_imagen(src, dst)
            if ok:
                clase_ok += 1
                # Guardar algunas para la muestra visual
                if len([m for m in muestras if m[2] == clase]) < 2:
                    muestras.append((src, dst, clase))
            else:
                total_fail += 1

        total_ok += clase_ok
        print(f"  ✅ {clase_ok}/{len(archivos)} procesadas")

    print(f"\n{'='*60}")
    print(f"  TOTAL procesadas: {total_ok}")
    print(f"  Fallidas:         {total_fail}")
    print(f"  Dataset listo en: {OUTPUT_DIR}")
    print(f"{'='*60}")

    # Generar muestra visual
    if muestras:
        generar_muestra(muestras)

    print("""
SIGUIENTE PASO:
  1. Revisa skull_strip_muestra.png para verificar calidad
  2. Si el resultado es bueno, actualiza DATA_DIR en preparacion_DS.py
     a: /home/davfy/Escritorio/Data_stripped/
  3. Vuelve a correr preparacion_DS.py con el nuevo origen
""")

if __name__ == "__main__":
    main()
