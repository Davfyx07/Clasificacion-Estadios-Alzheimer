"""
comparar_modelos_v2.py
Genera una comparativa visual y tabular de los 3 modelos entrenados.
Busca los archivos 'resumen.json' en la carpeta de resultados v2.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── CONFIGURACIÓN ──────────────────────────────────────────
BASE_RESULTS = "/home/davfy/Escritorio/Vision/v2/resultados"
OUTPUT_PATH  = os.path.join(BASE_RESULTS, "comparativa_modelos_v2.png")
CSV_PATH     = os.path.join(BASE_RESULTS, "metricas_comparativas_v2.csv")

MODELOS = [
    "EfficientNetV2S_CBAM_v2",
    "MobileNetV3_CBAM_v2",
    "ResNet50_CBAM_v2"
]

def cargar_resultados():
    data = []
    for m in MODELOS:
        ruta_json = os.path.join(BASE_RESULTS, m, "resumen.json")
        if os.path.exists(ruta_json):
            with open(ruta_json, 'r') as f:
                res = json.load(f)
                data.append(res)
        else:
            print(f"⚠️ Advertencia: No se encontró resumen para {m}")
    return data

def generar_comparativa():
    resultados = cargar_resultados()
    if not resultados:
        print("❌ No hay resultados para comparar. Asegúrate de haber entrenado los modelos.")
        return

    df = pd.DataFrame(resultados)
    
    # Guardar tabla en CSV para Excel
    df.to_csv(CSV_PATH, index=False)
    print(f"✅ Tabla comparativa guardada en: {CSV_PATH}")

    # --- GENERAR GRÁFICA ---
    # Convertir a formato largo para Seaborn
    df_plot = df.melt(id_vars="modelo", value_vars=["accuracy", "macro_f1", "balanced_acc"], 
                      var_name="Metrica", value_name="Valor")

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(data=df_plot, x="modelo", y="Valor", hue="Metrica", palette="viridis")
    
    # Añadir los valores encima de las barras
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=10, fontweight='bold')

    plt.title("Comparativa de Modelos V2 (35k imágenes)", fontsize=15, pad=20)
    plt.ylim(0, 1.1)
    plt.ylabel("Puntuación (0-1)")
    plt.xlabel("Arquitectura de Red Neuronal")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    print(f"✅ Gráfica comparativa guardada en: {OUTPUT_PATH}")

if __name__ == "__main__":
    generar_comparativa()