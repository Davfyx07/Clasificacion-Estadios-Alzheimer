"""
train_mobilenet.py
Entrenamiento preliminar MobileNetV3Large + CBAM
Versión de comparación — 10 épocas fase 1, 15 épocas fase 2
"""

import os, time, json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, balanced_accuracy_score)
from cbam import CBAM

# ── Configuración ──────────────────────────────────────────
MODEL_NAME  = "MobileNetV3_CBAM"
DATASET_DIR = "/home/davfy/Escritorio/Vision/dataset_balanceado"
RESULT_DIR  = f"/home/davfy/Escritorio/Vision/resultados/{MODEL_NAME}"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 32
SEED        = 42
EPOCAS_F1   = 10
EPOCAS_F2   = 15
LR_F1       = 1e-3
LR_F2       = 1e-4

CLASES = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]
MEAN   = [0.485, 0.456, 0.406]
STD    = [0.229, 0.224, 0.225]

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(RESULT_DIR, exist_ok=True)

# ── Modelo ─────────────────────────────────────────────────
class MobileNetCBAM(nn.Module):
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

# ── Dataset ────────────────────────────────────────────────
def get_loaders():
    t_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    t_eval = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    logica = {"Non Demented": 0, "Very mild Dementia": 1,
              "Mild Dementia": 2, "Moderate Dementia": 3}

    def remap(ds):
        m = {i: logica[c] for i, c in enumerate(ds.classes)}
        ds.targets = [m[t] for t in ds.targets]
        ds.samples = [(p, m[l]) for p, l in ds.samples]
        ds.class_to_idx = logica

    train_ds = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), t_train)
    val_ds   = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"),   t_eval)
    test_ds  = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"),  t_eval)

    for ds in [train_ds, val_ds, test_ds]:
        remap(ds)

    targets = np.array(train_ds.targets)
    counts  = np.bincount(targets)
    counts  = np.where(counts == 0, 1, counts)
    w       = 1.0 / counts
    sw      = torch.tensor([w[t] for t in targets], dtype=torch.double)
    sampler = WeightedRandomSampler(sw, len(sw))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader

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

    avg_loss = total_loss / total
    acc      = correct / total
    f1       = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return avg_loss, acc, f1, y_true, y_pred

# ── Guardar curvas ─────────────────────────────────────────
def guardar_curvas(history):
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(RESULT_DIR, "history.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{MODEL_NAME} — Curvas de Entrenamiento", fontsize=13, fontweight="bold")

    axes[0].plot(df["epoch"], df["train_loss"], label="Train")
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val")
    axes[0].axvline(x=EPOCAS_F1, color="gray", linestyle="--", alpha=0.5, label="Fase 2")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Época")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_acc"], label="Train")
    axes[1].plot(df["epoch"], df["val_acc"],   label="Val")
    axes[1].axvline(x=EPOCAS_F1, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Época")
    axes[1].legend()

    axes[2].plot(df["epoch"], df["val_f1"], color="green", label="Val Macro F1")
    axes[2].axvline(x=EPOCAS_F1, color="gray", linestyle="--", alpha=0.5, label="Fase 2")
    axes[2].set_title("Macro F1-Score")
    axes[2].set_xlabel("Época")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "curvas_entrenamiento.png"), dpi=150)
    plt.close()

# ── Matriz de confusión ────────────────────────────────────
def guardar_confusion(y_true, y_pred, titulo):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"{MODEL_NAME} — {titulo}", fontsize=13, fontweight="bold")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASES, yticklabels=CLASES, ax=axes[0])
    axes[0].set_title("Conteos absolutos")
    axes[0].set_xlabel("Predicción")
    axes[0].set_ylabel("Real")
    axes[0].tick_params(axis="x", rotation=30)

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASES, yticklabels=CLASES, ax=axes[1])
    axes[1].set_title("Normalizada (por fila)")
    axes[1].set_xlabel("Predicción")
    axes[1].set_ylabel("Real")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    nombre = titulo.lower().replace(" ", "_")
    plt.savefig(os.path.join(RESULT_DIR, f"confusion_{nombre}.png"), dpi=150)
    plt.close()

# ── Entrenamiento principal ────────────────────────────────
def entrenar():
    t_inicio = time.time()
    print(f"\nDispositivo: {DEVICE}")
    print(f"Modelo: {MODEL_NAME}")

    train_loader, val_loader, test_loader = get_loaders()
    print(f"Train: {len(train_loader.dataset)} imgs | "
          f"Val: {len(val_loader.dataset)} imgs | "
          f"Test: {len(test_loader.dataset)} imgs\n")

    model     = MobileNetCBAM().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales:     {total:,}")
    print(f"Parámetros entrenables: {trainable:,}")

    history    = []
    best_f1    = 0.0
    best_state = None

    # ── FASE 1 ────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"FASE 1 — CBAM + cabezal (backbone congelado)")
    print(f"{'='*55}")
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.cbam.parameters():
        p.requires_grad = True

    optimizer1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_F1, weight_decay=1e-5)

    for ep in range(EPOCAS_F1):
        t0 = time.time()
        tr_loss, tr_acc, _, _, _ = run_epoch(model, train_loader, criterion, optimizer1)
        vl_loss, vl_acc, vl_f1, _, _ = run_epoch(model, val_loader, criterion)
        elapsed = time.time() - t0
        history.append({"epoch": ep, "fase": 1,
                         "train_loss": tr_loss, "val_loss": vl_loss,
                         "train_acc": tr_acc,   "val_acc": vl_acc,
                         "val_f1": vl_f1})
        if vl_f1 > best_f1:
            best_f1    = vl_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  Época {ep+1:2d}/{EPOCAS_F1} | "
              f"Loss {tr_loss:.4f}/{vl_loss:.4f} | "
              f"Acc {tr_acc:.4f}/{vl_acc:.4f} | "
              f"F1 {vl_f1:.4f} | {elapsed:.1f}s")

    # ── FASE 2 ────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"FASE 2 — Fine-tuning (últimas capas + CBAM + cabezal)")
    print(f"{'='*55}")
    torch.cuda.empty_cache()

    # Descongelar últimas 5 capas del backbone
    for p in list(model.features.parameters())[-30:]:
        p.requires_grad = True

    trainable2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables Fase 2: {trainable2:,}\n")

    optimizer2  = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_F2, weight_decay=1e-5)
    scheduler2  = ReduceLROnPlateau(optimizer2, mode="max", factor=0.5,
                                     patience=3, verbose=True)
    patience    = 0
    MAX_PATIENCE = 7

    for ep in range(EPOCAS_F2):
        t0 = time.time()
        tr_loss, tr_acc, _, _, _ = run_epoch(model, train_loader, criterion, optimizer2)
        vl_loss, vl_acc, vl_f1, _, _ = run_epoch(model, val_loader, criterion)
        elapsed = time.time() - t0
        scheduler2.step(vl_f1)
        history.append({"epoch": EPOCAS_F1 + ep, "fase": 2,
                         "train_loss": tr_loss, "val_loss": vl_loss,
                         "train_acc": tr_acc,   "val_acc": vl_acc,
                         "val_f1": vl_f1})
        if vl_f1 > best_f1:
            best_f1    = vl_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1
        print(f"  Época {ep+1:2d}/{EPOCAS_F2} | "
              f"Loss {tr_loss:.4f}/{vl_loss:.4f} | "
              f"Acc {tr_acc:.4f}/{vl_acc:.4f} | "
              f"F1 {vl_f1:.4f} | {elapsed:.1f}s")
        if patience >= MAX_PATIENCE:
            print(f"  Early stopping en época {ep+1}")
            break

    # ── Evaluación final ──────────────────────────────────
    print(f"\n{'='*55}")
    print(f"EVALUACIÓN FINAL — TEST")
    print(f"{'='*55}")
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(RESULT_DIR, "best_model.pth"))

    _, _, _, y_true, y_pred = run_epoch(model, test_loader, criterion)

    macro_f1 = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    acc      = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    t_total  = (time.time() - t_inicio) / 60

    print(classification_report(y_true, y_pred, target_names=CLASES, zero_division=0))
    print(f"Macro F1-Score:     {macro_f1:.4f}")
    print(f"Balanced Accuracy:  {bal_acc:.4f}")
    print(f"Accuracy global:    {acc:.4f}")
    print(f"Tiempo total:       {t_total:.1f} min")

    guardar_curvas(history)
    guardar_confusion(y_true, y_pred, "Matriz de Confusión Test")

    resumen = {
        "modelo":       MODEL_NAME,
        "macro_f1":     round(macro_f1, 4),
        "balanced_acc": round(bal_acc, 4),
        "accuracy":     round(acc, 4),
        "mejor_f1_val": round(best_f1, 4),
        "tiempo_min":   round(t_total, 1),
        "parametros":   total,
        "epocas_f1":    EPOCAS_F1,
        "epocas_f2":    EPOCAS_F2,
        "batch_size":   BATCH_SIZE,
    }
    with open(os.path.join(RESULT_DIR, "resumen_final.json"), "w") as f:
        json.dump(resumen, f, indent=2)

    print(f"\n✅ Resultados guardados en: {RESULT_DIR}")
    return resumen

if __name__ == "__main__":
    entrenar()