import os
import subprocess
import sys

BASE = "/home/davfy/Escritorio/Vision"
V2_DIR = os.path.join(BASE, "v2")
TRAIN_ENGINE = os.path.join(V2_DIR, "train_engine.py")

if __name__ == "__main__":
    try:
        print("\nLanzando entrenamiento centralizado (train_engine.py)...")
        subprocess.run([sys.executable, TRAIN_ENGINE], check=True)
        print("\n✅ ENTRENAMIENTO COMPLETO. Revisa la carpeta v2/resultados")
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
