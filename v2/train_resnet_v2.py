"""
train_resnet_v2.py
ResNet50 + CBAM — versión corregida
CORRECCIONES v2:
- dataset_v2 sin augmentation doble
- LR_F2 reducido a 5e-5
- MAX_PATIENCE aumentado a 8
"""

import os, time, json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
import torch.nn as nn
from torchvision import models
from v2.dataset_v2 import get_dataloaders
from cbam import CBAM

MODEL_NAME  = "ResNet50_CBAM_v2"
RESULT_DIR  = f"/home/davfy/Escritorio/Vision/v2/resultados/{MODEL_NAME}"
CLASES      = ["Non_Demented", "Very_Mild_Demented", "Mild_Demented", "Moderate_Demented"]
os.makedirs(RESULT_DIR, exist_ok=True)

# ── Modelo ─────────────────────────────────────────────────
class ResNet50CBAM_v2(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base = models.resnet50(weights="IMAGENET1K_V2")
        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.classifier(x)

# ── Epoch ──────────────────────────────────────────────────
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

    return (total_loss / total,
            correct / total,
            f1_score(y_true, y_pred, average="macro", zero_division=0),
            y_true, y_pred)

def guardar_curvas(history):
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(RESULT_DIR, "history.csv"), index=False)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{MODEL_NAME} — Curvas de Entrenamiento", fontsize=13, fontweight="bold")
    axes[0].plot(df["epoch"], df["train_loss"], label="Train")
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val")
    axes[0].axvline(x=EPOCAS_F1, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(df["epoch"], df["train_acc"], label="Train")
    axes[1].plot(df["epoch"], df["val_acc"],   label="Val")
    axes[1].axvline(x=EPOCAS_F1, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(df["epoch"], df["val_f1"], color="green", label="Val Macro F1")
    axes[2].axvline(x=EPOCAS_F1, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_title("Macro F1"); axes[2].legend(); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "curvas_entrenamiento.png"), dpi=150)
    plt.close()

def guardar_confusion(y_true, y_pred, titulo):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"{MODEL_NAME} — {titulo}", fontsize=13, fontweight="bold")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASES, yticklabels=CLASES, ax=axes[0])
    axes[0].set_title("Conteos absolutos")
    axes[0].set_xlabel("Predicción"); axes[0].set_ylabel("Real")
    axes[0].tick_params(axis="x", rotation=30)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASES, yticklabels=CLASES, ax=axes[1])
    axes[1].set_title("Normalizada")
    axes[1].set_xlabel("Predicción"); axes[1].set_ylabel("Real")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR,
                f"confusion_{titulo.lower().replace(' ','_')}.png"), dpi=150)
    plt.close()

def entrenar():
    t_inicio = time.time()
    print(f"\nDispositivo: {DEVICE}\nModelo: {MODEL_NAME}")
    train_loader, val_loader, test_loader, _ = get_dataloaders(BATCH_SIZE)
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}\n")

    model     = ResNet50CBAM_v2().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    total     = sum(p.numel() for p in model.parameters())
    print(f"Parámetros totales: {total:,}")

    history = []; best_f1 = 0.0; best_state = None

    # ── FASE 1 ────────────────────────────────────────────
    print(f"\n{'='*55}\nFASE 1 — CBAM + cabezal (backbone congelado)\n{'='*55}")
    for p in model.stem.parameters():   p.requires_grad = False
    for p in model.layer1.parameters(): p.requires_grad = False
    for p in model.layer2.parameters(): p.requires_grad = False
    for p in model.layer3.parameters(): p.requires_grad = False
    for p in model.layer4.parameters(): p.requires_grad = False
    for p in model.layer3[1].parameters(): p.requires_grad = True
    for p in model.layer4[1].parameters(): p.requires_grad = True

    trainable1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables Fase 1: {trainable1:,}\n")

    optimizer1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_F1, weight_decay=1e-5)

    for ep in range(EPOCAS_F1):
        t0 = time.time()
        tr_loss, tr_acc, _, _, _ = run_epoch(model, train_loader, criterion, optimizer1)
        vl_loss, vl_acc, vl_f1, _, _ = run_epoch(model, val_loader, criterion)
        history.append({"epoch": ep, "fase": 1,
                        "train_loss": tr_loss, "val_loss": vl_loss,
                        "train_acc":  tr_acc,  "val_acc":  vl_acc, "val_f1": vl_f1})
        if vl_f1 > best_f1:
            best_f1 = vl_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  Época {ep+1:2d}/{EPOCAS_F1} | Loss {tr_loss:.4f}/{vl_loss:.4f} | "
              f"Acc {tr_acc:.4f}/{vl_acc:.4f} | F1 {vl_f1:.4f} | {time.time()-t0:.1f}s")

    # ── FASE 2 ────────────────────────────────────────────
    print(f"\n{'='*55}\nFASE 2 — Fine-tuning (layer3 + layer4 + CBAM + cabezal)\n{'='*55}")
    torch.cuda.empty_cache()
    for p in model.layer3.parameters(): p.requires_grad = True
    for p in model.layer4.parameters(): p.requires_grad = True

    trainable2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables Fase 2: {trainable2:,}\n")

    optimizer2 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_F2, weight_decay=1e-5)
    scheduler2 = ReduceLROnPlateau(optimizer2, mode="max", factor=0.5, patience=4)
    patience = 0; MAX_PATIENCE = 8

    for ep in range(EPOCAS_F2):
        t0 = time.time()
        tr_loss, tr_acc, _, _, _ = run_epoch(model, train_loader, criterion, optimizer2)
        vl_loss, vl_acc, vl_f1, _, _ = run_epoch(model, val_loader, criterion)
        scheduler2.step(vl_f1)
        history.append({"epoch": EPOCAS_F1 + ep, "fase": 2,
                        "train_loss": tr_loss, "val_loss": vl_loss,
                        "train_acc":  tr_acc,  "val_acc":  vl_acc, "val_f1": vl_f1})
        if vl_f1 > best_f1:
            best_f1 = vl_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        print(f"  Época {ep+1:2d}/{EPOCAS_F2} | Loss {tr_loss:.4f}/{vl_loss:.4f} | "
              f"Acc {tr_acc:.4f}/{vl_acc:.4f} | F1 {vl_f1:.4f} | {time.time()-t0:.1f}s")
        if patience >= MAX_PATIENCE:
            print(f"  Early stopping en época {ep+1}"); break

    # ── Evaluación final ──────────────────────────────────
    print(f"\n{'='*55}\nEVALUACIÓN FINAL — TEST\n{'='*55}")
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(RESULT_DIR, "best_model.pth"))
    _, _, _, y_true, y_pred = run_epoch(model, test_loader, criterion)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    acc      = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    t_total  = (time.time() - t_inicio) / 60

    print(classification_report(y_true, y_pred, target_names=CLASES, zero_division=0))
    print(f"Macro F1: {macro_f1:.4f} | Balanced Acc: {bal_acc:.4f} | Acc: {acc:.4f} | Tiempo: {t_total:.1f}min")

    guardar_curvas(history)
    guardar_confusion(y_true, y_pred, "Matriz de Confusión Test")

    resumen = {"modelo": MODEL_NAME, "macro_f1": round(macro_f1, 4),
               "balanced_acc": round(bal_acc, 4), "accuracy": round(acc, 4),
               "mejor_f1_val": round(best_f1, 4), "tiempo_min": round(t_total, 1),
               "parametros": total, "epocas_f1": EPOCAS_F1, "epocas_f2": EPOCAS_F2,
               "batch_size": BATCH_SIZE}
    with open(os.path.join(RESULT_DIR, "resumen_final.json"), "w") as f:
        json.dump(resumen, f, indent=2)
    print(f"\n✅ Resultados en: {RESULT_DIR}")
    return resumen

if __name__ == "__main__":
    entrenar()