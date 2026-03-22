import os, sys, time, subprocess

BASE    = "/home/davfy/Escritorio/Vision"
V2_DIR  = os.path.join(BASE, "v2")

def correr(script):
    print(f"\nLanzando {script}...")
    subprocess.run([sys.executable, os.path.join(V2_DIR, script)], check=True)

try:
    correr("train_efficientnet_v2.py")
    correr("train_mobilenet_v2.py")
    correr("train_resnet_v2.py")
    print("\n✅ ENTRENAMIENTO COMPLETO. Revisa la carpeta v2/resultados")
except Exception as e:
    print(f"\n❌ Error durante la ejecución: {e}")