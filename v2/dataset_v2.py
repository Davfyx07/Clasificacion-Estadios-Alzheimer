import os
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

DATASET_DIR = "/home/davfy/Escritorio/Vision/dataset_balanceado"
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Transformaciones: Solo normalización, el aumento ya está en disco
transform_generico = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

def get_dataloaders(batch_size=32, num_workers=4):
    train_ds = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform_generico)
    val_ds   = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"),   transform_generico)
    test_ds  = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"),  transform_generico)

    # SINCRONIZACIÓN CON TUS NUEVAS CARPETAS
    logica_medica = {
        "Non_Demented":       0,
        "Very_Mild_Demented": 1,
        "Mild_Demented":      2,
        "Moderate_Demented":  3,
    }

    def remap(ds):
        # Mapea los índices alfabéticos al orden médico real
        m = {i: logica_medica[c] for i, c in enumerate(ds.classes)}
        ds.targets = [m[t] for t in ds.targets]
        ds.samples = [(p, m[l]) for p, l in ds.samples]
        ds.class_to_idx = logica_medica

    for ds in [train_ds, val_ds, test_ds]:
        remap(ds)

    # Sampler para asegurar balance en cada batch
    targets = np.array(train_ds.targets)
    counts  = np.bincount(targets)
    w       = 1.0 / counts
    sw      = torch.tensor([w[t] for t in targets], dtype=torch.double)
    sampler = WeightedRandomSampler(sw, len(sw))

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, list(logica_medica.keys())