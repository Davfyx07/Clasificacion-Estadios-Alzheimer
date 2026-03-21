import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración
resultados_dir = "resultados"
modelos = ["EfficientNetV2S_CBAM", "MobileNetV3_CBAM", "ResNet50_CBAM"]
data = []

print("--- Iniciando recolección desde history.csv (Columnas: val_acc, val_f1) ---")

for m in modelos:
    csv_path = os.path.join(resultados_dir, m, "history.csv")
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            
            # Filtramos solo la Fase 2 (Fine-tuning) para comparar el rendimiento final
            fase2_df = df[df['fase'] == 2]
            
            # Si no hay fase 2 (porque falló o no llegó), usamos todo el df
            target_df = fase2_df if not fase2_df.empty else df
            
            # Buscamos la fila con el mejor F1 de validación
            best_idx = target_df['val_f1'].idxmax()
            best_row = target_df.loc[best_idx]

            data.append({
                "Modelo": m,
                "Accuracy": best_row['val_acc'],
                "F1-Score": best_row['val_f1'],
                "Loss": best_row['val_loss'],
                "Mejor Epoca": best_idx + 1
            })
            print(f"✅ {m}: F1 {best_row['val_f1']:.4f} en época {best_idx + 1}")
        except Exception as e:
            print(f"⚠️ Error al procesar {m}: {e}")
    else:
        print(f"❌ No se encontró: {csv_path}")

if not data:
    print("\n‼️ ERROR: No se recolectaron datos. Verifica las rutas.")
else:
    df_final = pd.DataFrame(data)
    df_final.to_csv("comparativa_final.csv", index=False)
    
    # Gráfica
    plt.figure(figsize=(10, 6))
    df_melted = df_final.melt(id_vars="Modelo", value_vars=["Accuracy", "F1-Score"], 
                              var_name="Métrica", value_name="Valor")
    
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df_melted, x="Modelo", y="Valor", hue="Métrica", palette="viridis")
    
    # Añadir los valores encima de las barras
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')

    plt.title("Comparativa de Modelos: Alzheimer Classification (USCO)", fontsize=14)
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    plt.savefig("grafica_comparativa.png")
    print("\n🚀 ¡Éxito! Revisa 'comparativa_final.csv' y 'grafica_comparativa.png'")