"""
train_resnet_v2.py
ResNet50 + CBAM — versión corregida
CORRECCIONES v2:
- dataset_v2 sin augmentation doble
- LR_F2 reducido a 5e-5
- MAX_PATIENCE aumentado a 8
"""

import os, time, json, sys
# Asegurar rutas
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd  # <--- Crucial para guardar_curvas
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (classification_report, confusion_matrix, 
                              f1_score, balanced_accuracy_score)
from cbam import CBAM
from v2.dataset_v2 import get_dataloaders

# ── CONFIGURACIÓN V2 ──────────────────────────────────────────
MODEL_NAME  = "ResNet50_CBAM_v2"
RESULT_DIR  = f"/home/davfy/Escritorio/Vision/v2/resultados/{MODEL_NAME}"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASES      = ["Non_Demented", "Very_Mild_Demented", "Mild_Demented", "Moderate_Demented"]

# ── VARIABLES DE CONTROL ──────────────────────────────────────
BATCH_SIZE    = 32
SEED          = 42
EPOCAS_F1     = 10   # Fase 1: Solo CBAM + Cabeza
EPOCAS_F2     = 20   # Fase 2: Fine-tuning Layer 3 y 4
LR_F1         = 1e-3
LR_F2         = 5e-5
MAX_PATIENCE  = 8

os.makedirs(RESULT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Modelo ─────────────────────────────────────────────────
class ResNet50CBAM_v2(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Usamos V2 de pesos para mejor punto de partida
        base = models.resnet50(weights="IMAGENET1K_V2")
        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        # Inyectamos CBAM después de los bloques residuales
        self.layer3 = nn.Sequential(base.layer3, CBAM(1024))
        self.layer4 = nn.Sequential(base.layer4, CBAM(2048))
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
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

# ── Visualización ──────────────────────────────────────────
def guardar_curvas(history):
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(RESULT_DIR, "history.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(df["epoch"], df["train_loss"], label="Train")
    axes[0].plot(df["epoch"], df["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].legend()
    
    axes[1].plot(df["epoch"], df["val_f1"], color="green", label="Val Macro F1")
    axes[1].set_title("Macro F1-Score"); axes[1].legend()
    
    plt.savefig(os.path.join(RESULT_DIR, "curvas.png")); plt.close()

def guardar_confusion(y_true, y_pred, titulo):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASES, yticklabels=CLASES)
    plt.title(titulo); plt.ylabel("Real"); plt.xlabel("Predicción")
    plt.savefig(os.path.join(RESULT_DIR, f"confusion_final.png")); plt.close()

# ── Entrenamiento ──────────────────────────────────────────
def entrenar():
    t_inicio = time.time()
    train_loader, val_loader, test_loader, _ = get_dataloaders(BATCH_SIZE)
    model = ResNet50CBAM_v2().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    history = []; best_f1 = 0.0; best_state = None

    # FASE 1: Congelar backbone
    for p in model.stem.parameters(): p.requires_grad = False
    for p in model.layer1.parameters(): p.requires_grad = False
    for p in model.layer2.parameters(): p.requires_grad = False
    for p in model.layer3[0].parameters(): p.requires_grad = False
    for p in model.layer4[0].parameters(): p.requires_grad = False
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_F1)
    
    print(f"--- FASE 1: Iniciando {MODEL_NAME} ---")
    for ep in range(EPOCAS_F1):
        tr_l, tr_a, _, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        vl_l, vl_a, vl_f, _, _ = run_epoch(model, val_loader, criterion)
        history.append({"epoch": ep, "train_loss": tr_l, "val_loss": vl_l, "val_acc": vl_a, "val_f1": vl_f})
        print(f"E{ep+1:02d} | Val F1: {vl_f:.4f} | loss: {vl_l:.4f} | acc: {vl_a:.4f} | time: {(time.time() - t_inicio)/60:.2f} min")
        if vl_f > best_f1:
            best_f1 = vl_f
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # FASE 2: Fine-tuning
    print("--- FASE 2: Descongelando Layer 3 y 4 ---")
    for p in model.layer3.parameters(): p.requires_grad = True
    for p in model.layer4.parameters(): p.requires_grad = True
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_F2)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    patience_cnt = 0

    for ep in range(EPOCAS_F2):
        tr_l, tr_a, _, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        vl_l, vl_a, vl_f, _, _ = run_epoch(model, val_loader, criterion)
        scheduler.step(vl_f)
        history.append({"epoch": ep + EPOCAS_F1, "train_loss": tr_l, "val_loss": vl_l, "val_acc": vl_a, "val_f1": vl_f})
        print(f"E{ep+1+EPOCAS_F1:02d} | Val F1: {vl_f:.4f} | loss: {vl_l:.4f} | acc: {vl_a:.4f} | time: {(time.time() - t_inicio)/60:.2f} min")
        
        if vl_f > best_f1:
            best_f1 = vl_f
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= MAX_PATIENCE: break

    # Final
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(RESULT_DIR, "best_model.pth"))
    _, _, _, y_true, y_pred = run_epoch(model, test_loader, criterion)
    
    print("\n✅ ResNet50 Entrenamiento Completo")
    print(classification_report(y_true, y_pred, target_names=CLASES))
    guardar_curvas(history)
    guardar_confusion(y_true, y_pred, "Test Final")
    
    resumen = {"modelo": MODEL_NAME, "macro_f1": f1_score(y_true, y_pred, average="macro")}
    with open(os.path.join(RESULT_DIR, "resumen.json"), "w") as f:
        json.dump(resumen, f)

if __name__ == "__main__":
    entrenar()