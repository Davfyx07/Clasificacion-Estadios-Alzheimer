"""
train_efficientnet_v2.py
EfficientNetV2S + CBAM — versión corregida
CORRECCIONES v2:
- Fase 1 congela backbone correctamente: solo entrenan CBAM + cabezal (~700k params)
- dataset_v2 sin augmentation doble
- LR_F2 reducido a 5e-5 para evitar overfitting en fine-tuning
- MAX_PATIENCE aumentado para dar más tiempo al modelo
"""

import os, time, json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, balanced_accuracy_score)
from cbam import CBAM
from v2.dataset_v2 import get_dataloaders

# CONFIGURACIÓN V2
MODEL_NAME  = "EfficientNetV2S_CBAM_v2"
RESULT_DIR  = f"/home/davfy/Escritorio/Vision/v2/resultados/{MODEL_NAME}"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASES      = ["Non_Demented", "Very_Mild_Demented", "Mild_Demented", "Moderate_Demented"]
# ── VARIABLES DE CONTROL (Faltantes en tu versión) ──────────
BATCH_SIZE    = 32
SEED          = 42
EPOCAS_F1     = 10   # Fase 1: Solo entrenamiento de CBAM y Cabezal
EPOCAS_F2     = 20   # Fase 2: Fine-tuning del bloque 4 al 6
LR_F1         = 1e-3
LR_F2         = 5e-5
MAX_PATIENCE  = 8
os.makedirs(RESULT_DIR, exist_ok=True)
# Configurar semilla para reproducibilidad
torch.manual_seed(SEED)
np.random.seed(SEED)
# ── Modelo ─────────────────────────────────────────────────
class EfficientNetCBAM_v2(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        self.features = base.features
        self._insert_cbam()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def _insert_cbam(self):
        # Insertamos CBAM en los bloques finales donde la semántica es más rica
        for idx, ch in {4: 128, 5: 160, 6: 256}.items():
            self.features[idx] = nn.Sequential(self.features[idx], CBAM(ch))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

# ── Epoch train/eval ───────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0, 0, 0
    y_true, y_pred = [], []

    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out  = model(imgs)
            loss = criterion(out, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return (total_loss / total, correct / total, 
            f1_score(y_true, y_pred, average="macro", zero_division=0), 
            y_true, y_pred)

# ── Funciones de Guardado ──────────────────────────────────
def guardar_curvas(history):
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(RESULT_DIR, "history.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(df["epoch"], df["train_loss"], label="Train"); axes[0].plot(df["epoch"], df["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(df["epoch"], df["val_f1"], color="green", label="Val Macro F1")
    axes[1].set_title("Macro F1-Score"); axes[1].legend()
    plt.savefig(os.path.join(RESULT_DIR, "curvas.png")); plt.close()

def guardar_confusion(y_true, y_pred, titulo):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASES, yticklabels=CLASES)
    plt.title(titulo); plt.ylabel("Real"); plt.xlabel("Predicción")
    plt.savefig(os.path.join(RESULT_DIR, f"confusion_{titulo.lower().replace(' ', '_')}.png")); plt.close()

# ── Entrenamiento Principal ────────────────────────────────
def entrenar():
    t_inicio = time.time()
    train_loader, val_loader, test_loader, _ = get_dataloaders(BATCH_SIZE)
    model = EfficientNetCBAM_v2().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    best_f1 = 0.0
    best_state = None

    # FASE 1: Congelar Backbone, entrenar CBAM + Cabezal
    for p in model.features.parameters(): p.requires_grad = False
    for idx in [4, 5, 6]:
        for p in model.features[idx][1].parameters(): p.requires_grad = True
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_F1)
    
    print("--- INICIANDO FASE 1 ---")
    for ep in range(EPOCAS_F1):
        tr_loss, tr_acc, _, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, vl_f1, _, _ = run_epoch(model, val_loader, criterion)
        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": vl_loss, "val_acc": vl_acc, "val_f1": vl_f1})
        print(f"E{ep+1+EPOCAS_F1}/{EPOCAS_F1+EPOCAS_F2} | Val F1: {vl_f1:.4f} | loss: {vl_loss:.4f} | acc: {vl_acc:.4f} | time: {(time.time() - t_inicio)/60:.2f} min")

        if vl_f1 > best_f1:
            best_f1 = vl_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # FASE 2: Fine-tuning bloques finales
    print("--- INICIANDO FASE 2 ---")
    for idx in [4, 5, 6]:
        for p in model.features[idx].parameters(): p.requires_grad = True
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_F2)
    patience_cnt = 0

    for ep in range(EPOCAS_F2):
        tr_loss, tr_acc, _, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, vl_f1, _, _ = run_epoch(model, val_loader, criterion)
        history.append({"epoch": ep + EPOCAS_F1, "train_loss": tr_loss, "val_loss": vl_loss, "val_acc": vl_acc, "val_f1": vl_f1})
        print(f"E{ep+1+EPOCAS_F1}/{EPOCAS_F1+EPOCAS_F2} | Val F1: {vl_f1:.4f} | loss: {vl_loss:.4f} | acc: {vl_acc:.4f} | time: {(time.time() - t_inicio)/60:.2f} min")
        
        if vl_f1 > best_f1:
            best_f1 = vl_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= MAX_PATIENCE: break

    # Evaluación Final
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(RESULT_DIR, "best_model.pth"))
    _, _, _, y_true, y_pred = run_epoch(model, test_loader, criterion)
    
    print("\nREPORT TEST:")
    print(classification_report(y_true, y_pred, target_names=CLASES))
    guardar_curvas(history)
    guardar_confusion(y_true, y_pred, "Test Final")
    
    with open(os.path.join(RESULT_DIR, "resumen.json"), "w") as f:
        json.dump({"modelo": MODEL_NAME, "accuracy": balanced_accuracy_score(y_true, y_pred), "macro_f1": f1_score(y_true, y_pred, average="macro")}, f)

if __name__ == "__main__":
    entrenar()