import argparse
import json
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cbam import CBAM
from v2.dataset_v2 import get_dataloaders

CLASSES = ["Non_Demented", "Very_Mild_Demented", "Mild_Demented", "Moderate_Demented"]
BASE_RESULTS = "/home/davfy/Escritorio/Vision/v2/resultados"
DEFAULT_DATASET = "/home/davfy/Escritorio/Vision/dataset_balanceado"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, input_data, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_image, target_class):
        self.model.eval()
        output = self.model(input_image)
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    def save_visual(self, img_tensor, cam, path, title):
        img = img_tensor[0].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Original")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.imshow(cam, cmap="jet", alpha=0.5)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def close(self):
        for hook in self.hooks:
            hook.remove()


class EfficientNetCBAMV2(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        self.features = base.features
        for idx, ch in {4: 128, 5: 160, 6: 256}.items():
            self.features[idx] = nn.Sequential(self.features[idx], CBAM(ch))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class MobileNetCBAMV2(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
        self.features = base.features
        self.cbam = CBAM(960)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(960, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x)
        return self.classifier(x)


class ResNet50CBAMV2(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base = models.resnet50(weights="IMAGENET1K_V2")
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = nn.Sequential(base.layer3, CBAM(1024))
        self.layer4 = nn.Sequential(base.layer4, CBAM(2048))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.classifier(x)


MODEL_CONFIG = {
    "efficientnet": {
        "model_name": "EfficientNetV2S_CBAM_v2",
        "builder": EfficientNetCBAMV2,
        "epochs_f1": 10,
        "epochs_f2": 20,
        "lr_f1": 1e-3,
        "lr_f2": 5e-5,
        "patience": 8,
        "target_layer": lambda m: m.features[6][1],
        "freeze_f1": lambda m: [setattr(p, "requires_grad", False) for p in m.features.parameters()],
        "unfreeze_f2": lambda m: [setattr(p, "requires_grad", True) for idx in [4, 5, 6] for p in m.features[idx].parameters()],
    },
    "mobilenet": {
        "model_name": "MobileNetV3_CBAM_v2",
        "builder": MobileNetCBAMV2,
        "epochs_f1": 10,
        "epochs_f2": 15,
        "lr_f1": 1e-3,
        "lr_f2": 1e-4,
        "patience": 7,
        "target_layer": lambda m: m.cbam.spatial_attention.conv,
        "freeze_f1": lambda m: [setattr(p, "requires_grad", False) for p in m.features.parameters()],
        "unfreeze_f2": lambda m: [setattr(p, "requires_grad", True) for p in list(m.features.parameters())[-30:]],
    },
    "resnet": {
        "model_name": "ResNet50_CBAM_v2",
        "builder": ResNet50CBAMV2,
        "epochs_f1": 10,
        "epochs_f2": 20,
        "lr_f1": 1e-3,
        "lr_f2": 5e-5,
        "patience": 8,
        "target_layer": lambda m: m.layer4[1].spatial_attention.conv,
        "freeze_f1": lambda m: (
            [setattr(p, "requires_grad", False) for p in m.stem.parameters()]
            + [setattr(p, "requires_grad", False) for p in m.layer1.parameters()]
            + [setattr(p, "requires_grad", False) for p in m.layer2.parameters()]
            + [setattr(p, "requires_grad", False) for p in m.layer3[0].parameters()]
            + [setattr(p, "requires_grad", False) for p in m.layer4[0].parameters()]
        ),
        "unfreeze_f2": lambda m: (
            [setattr(p, "requires_grad", True) for p in m.layer3.parameters()]
            + [setattr(p, "requires_grad", True) for p in m.layer4.parameters()]
        ),
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer=None, phase="train"):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, total = 0.0, 0
    y_true, y_pred = [], []
    pbar = tqdm(loader, desc=f"[{phase}] {len(loader)} batches", leave=False)
    with torch.set_grad_enabled(training):
        for imgs, labels in pbar:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            preds = outputs.argmax(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    return total_loss / max(total, 1), macro_f1, bal_acc, y_true, y_pred


def save_curves(history, output_dir):
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(output_dir, "history.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(df["epoch"], df["train_loss"], label="Train")
    axes[0].plot(df["epoch"], df["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(df["epoch"], df["val_f1"], label="Val Macro F1", color="green")
    axes[1].set_title("Macro F1")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "curvas.png"))
    plt.close()


def save_confusion(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.ylabel("Real")
    plt.xlabel("Prediccion")
    plt.title("Matriz de confusion - Test")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_test.png"))
    plt.close()


def save_gradcam(model, target_layer, test_loader, output_dir):
    gradcam = GradCAM(model, target_layer)
    class_done = set()
    for imgs, labels in test_loader:
        for class_idx in range(len(CLASSES)):
            if class_idx in class_done:
                continue
            idx = (labels == class_idx).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                continue
            img = imgs[idx[0] : idx[0] + 1].to(DEVICE)
            cam = gradcam.generate(img, class_idx)
            gradcam.save_visual(
                img,
                cam,
                os.path.join(output_dir, f"gradcam_class_{class_idx}.png"),
                f"Atencion: {CLASSES[class_idx]}",
            )
            class_done.add(class_idx)
        if len(class_done) == len(CLASSES):
            break
    gradcam.close()


def load_dataloaders(dataset_dir, batch_size):
    loaders = get_dataloaders(dataset_dir, batch_size=batch_size)
    if len(loaders) == 4:
        train_loader, val_loader, test_loader, _ = loaders
    else:
        train_loader, val_loader, test_loader = loaders
    return train_loader, val_loader, test_loader


def train_one_model(model_key, train_loader, val_loader, test_loader, use_gradcam):
    cfg = MODEL_CONFIG[model_key]
    model_name = cfg["model_name"]
    out_dir = os.path.join(BASE_RESULTS, model_name)
    os.makedirs(out_dir, exist_ok=True)

    model = cfg["builder"](num_classes=len(CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_f1 = -1.0
    best_state = None
    start = time.time()
    print(f"\n=== Entrenando {model_name} ===")

    cfg["freeze_f1"](model)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr_f1"])
    for ep in range(cfg["epochs_f1"]):
        tr_loss, _, _, _, _ = run_epoch(model, train_loader, criterion, optimizer, phase="train-f1")
        vl_loss, vl_f1, vl_bacc, _, _ = run_epoch(model, val_loader, criterion, phase="val-f1")
        epoch_global = ep + 1
        history.append(
            {"epoch": epoch_global, "train_loss": tr_loss, "val_loss": vl_loss, "val_f1": vl_f1, "val_balanced_acc": vl_bacc}
        )
        print(f"F1 E{epoch_global:02d}: val_f1={vl_f1:.4f} val_bacc={vl_bacc:.4f}")
        if vl_f1 > best_f1:
            best_f1 = vl_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    cfg["unfreeze_f2"](model)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr_f2"])
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    wait = 0
    for ep in range(cfg["epochs_f2"]):
        tr_loss, _, _, _, _ = run_epoch(model, train_loader, criterion, optimizer, phase="train-f2")
        vl_loss, vl_f1, vl_bacc, _, _ = run_epoch(model, val_loader, criterion, phase="val-f2")
        scheduler.step(vl_f1)
        epoch_global = cfg["epochs_f1"] + ep + 1
        history.append(
            {"epoch": epoch_global, "train_loss": tr_loss, "val_loss": vl_loss, "val_f1": vl_f1, "val_balanced_acc": vl_bacc}
        )
        print(f"F2 E{epoch_global:02d}: val_f1={vl_f1:.4f} val_bacc={vl_bacc:.4f}")
        if vl_f1 > best_f1:
            best_f1 = vl_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                print("Early stopping (fase 2)")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(out_dir, "best_model.pth"))
    ts_loss, ts_f1, ts_bacc, y_true, y_pred = run_epoch(model, test_loader, criterion, phase="test")
    report = classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0)

    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    save_curves(history, out_dir)
    save_confusion(y_true, y_pred, out_dir)
    if use_gradcam:
        save_gradcam(model, cfg["target_layer"](model), test_loader, out_dir)

    summary = {
        "modelo": model_name,
        "test_loss": round(ts_loss, 4),
        "accuracy": round(float(np.mean(np.array(y_true) == np.array(y_pred))), 4),
        "balanced_acc": round(ts_bacc, 4),
        "macro_f1": round(ts_f1, 4),
        "best_val_f1": round(best_f1, 4),
        "minutes": round((time.time() - start) / 60, 2),
    }
    with open(os.path.join(out_dir, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✅ {model_name} finalizado | test_f1={summary['macro_f1']:.4f}")
    return summary


def save_global_comparison(all_summaries):
    os.makedirs(BASE_RESULTS, exist_ok=True)
    df = pd.DataFrame(all_summaries)
    csv_path = os.path.join(BASE_RESULTS, "metricas_comparativas_v2.csv")
    fig_path = os.path.join(BASE_RESULTS, "comparativa_modelos_v2.png")
    df.to_csv(csv_path, index=False)

    plot_df = df.melt(id_vars="modelo", value_vars=["accuracy", "balanced_acc", "macro_f1"], var_name="metrica", value_name="valor")
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=plot_df, x="modelo", y="valor", hue="metrica", palette="viridis")
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2.0, h), ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")
    plt.ylim(0, 1.1)
    plt.title("Comparativa modelos v2")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"📊 Comparativa global guardada en: {csv_path} y {fig_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento centralizado de modelos Alzheimer MRI (v2)")
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET, help="Ruta a dataset con subcarpetas train/val/test")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Semilla global")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_CONFIG.keys()), default=["efficientnet", "mobilenet", "resnet"], help="Modelos a entrenar en orden")
    parser.add_argument("--no-gradcam", action="store_true", help="Desactivar exportación de Grad-CAM")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(BASE_RESULTS, exist_ok=True)
    print(f"Dispositivo: {DEVICE}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Orden modelos: {args.models}")
    train_loader, val_loader, test_loader = load_dataloaders(args.dataset_dir, args.batch_size)
    summaries = []
    for model_key in args.models:
        summaries.append(train_one_model(model_key, train_loader, val_loader, test_loader, use_gradcam=not args.no_gradcam))
    save_global_comparison(summaries)
    print("🏁 Entrenamiento centralizado completado.")


if __name__ == "__main__":
    main()
