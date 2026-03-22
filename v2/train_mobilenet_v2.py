import os, time, json, sys
# Asegurar que reconozca el módulo v2 y cbam desde la raíz
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (classification_report, confusion_matrix, 
                              f1_score, balanced_accuracy_score)
from cbam import CBAM
from v2.dataset_v2 import get_dataloaders # Usamos el centralizado

# ── CONFIGURACIÓN V2 ──────────────────────────────────────────
MODEL_NAME  = "MobileNetV3_CBAM_v2"
RESULT_DIR  = f"/home/davfy/Escritorio/Vision/v2/resultados/{MODEL_NAME}"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASES      = ["Non_Demented", "Very_Mild_Demented", "Mild_Demented", "Moderate_Demented"]

# VARIABLES DE CONTROL
BATCH_SIZE    = 32
SEED          = 42
EPOCAS_F1     = 10   # Fase 1: Solo entrenamiento de CBAM y Cabezal
EPOCAS_F2     = 15   # Fase 2: Fine-tuning de las últimas capas
LR_F1         = 1e-3
LR_F2         = 1e-4
MAX_PATIENCE  = 7

os.makedirs(RESULT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Modelo ─────────────────────────────────────────────────
class MobileNetCBAM_v2(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
        self.features = base.features  # 16 bloques
        # CBAM al final del backbone (960 canales)
        self.cbam = CBAM(960)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(960, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
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
    axes[0].plot(df["epoch"], df["train_loss"], label="Train")
    axes[0].plot(df["epoch"], df["val_loss"], label="Val")
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

# ── Entrenamiento principal ────────────────────────────────
def entrenar():
    t_inicio = time.time()
    print(f"\n🚀 Iniciando {MODEL_NAME}")
    
    # Importante: get_dataloaders viene de dataset_v2
    train_loader, val_loader, test_loader, _ = get_dataloaders(BATCH_SIZE)
    
    model = MobileNetCBAM_v2().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    best_f1 = 0.0
    best_state = None

    # FASE 1: Congelar backbone (solo CBAM y cabezal)
    for p in model.features.parameters():
        p.requires_grad = False
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_F1)
    
    print("\n--- FASE 1: Backbone Congelado ---")
    for ep in range(EPOCAS_F1):
        tr_loss, tr_acc, _, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, vl_f1, _, _ = run_epoch(model, val_loader, criterion)
        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": vl_loss, "val_acc": vl_acc, "val_f1": vl_f1})
        print(f"E{ep+1:02d} | Loss: {tr_loss:.4f} | Val F1: {vl_f1:.4f} | loss: {vl_loss:.4f} | acc: {vl_acc:.4f} | time: {(time.time() - t_inicio)/60:.2f} min")
        if vl_f1 > best_f1:
            best_f1 = vl_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # FASE 2: Fine-tuning (descongelar últimas capas)
    print("\n--- FASE 2: Fine-tuning ---")
    # Descongelar las últimas 30 capas de parámetros (aprox. los últimos bloques)
    for p in list(model.features.parameters())[-30:]:
        p.requires_grad = True
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_F2)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    patience_cnt = 0

    for ep in range(EPOCAS_F2):
        tr_loss, tr_acc, _, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, vl_f1, _, _ = run_epoch(model, val_loader, criterion)
        scheduler.step(vl_f1)
        history.append({"epoch": ep + EPOCAS_F1, "train_loss": tr_loss, "val_loss": vl_loss, "val_acc": vl_acc, "val_f1": vl_f1})
        print(f"E{ep+1+EPOCAS_F1:02d} | Loss: {tr_loss:.4f} | Val F1: {vl_f1:.4f}")
        
        if vl_f1 > best_f1:
            best_f1 = vl_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= MAX_PATIENCE:
                print("Early stopping!")
                break

    # Evaluación Final con el mejor modelo guardado
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(RESULT_DIR, "best_model.pth"))
    _, _, _, y_true, y_pred = run_epoch(model, test_loader, criterion)
    
    print("\n✅ Evaluación Final en Test:")
    print(classification_report(y_true, y_pred, target_names=CLASES))
    
    guardar_curvas(history)
    guardar_confusion(y_true, y_pred, "Test Final")
    
    # Resumen para el script de comparativa
    resumen = {
        "modelo": MODEL_NAME,
        "accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4)
    }
    with open(os.path.join(RESULT_DIR, "resumen.json"), "w") as f:
        json.dump(resumen, f)

if __name__ == "__main__":
    entrenar()