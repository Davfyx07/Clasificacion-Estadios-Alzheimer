import os
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from collections import Counter

# ── Configuración ──────────────────────────────────────────
DATASET_DIR = "/home/davfy/Escritorio/Vision/dataset_balanceado"
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ── Transformaciones con Aumentación (En RAM) ──────────────
# Esto ayuda a que el modelo no memorice al único paciente de Moderate
transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

transform_val_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])
def get_dataloaders(batch_size=32, num_workers=4):
    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir   = os.path.join(DATASET_DIR, "val")
    test_dir  = os.path.join(DATASET_DIR, "test")

    train_ds = datasets.ImageFolder(train_dir, transform=transform_train)
    val_ds   = datasets.ImageFolder(val_dir,   transform=transform_val_test)
    test_ds  = datasets.ImageFolder(test_dir,  transform=transform_val_test)

    # 1. Definimos el orden médico real
    logica_medica = {
        "Non Demented": 0,
        "Very mild Dementia": 1,
        "Mild Dementia": 2,
        "Moderate Dementia": 3
    }
    
    # 2. CIRUGÍA PROFUNDA: Reasignar targets Y samples
    for ds in [train_ds, val_ds, test_ds]:
        # Mapeo de lo que PyTorch leyó (alfabético) a lo que tú quieres (médico)
        old_classes = ds.classes # Ejemplo: ['Mild', 'Moderate', 'Non', 'Very mild']
        map_idx = {old_idx: logica_medica[name] for old_idx, name in enumerate(old_classes)}
        
        # Actualizamos targets
        ds.targets = [map_idx[t] for t in ds.targets]
        
        # Actualizamos samples (¡ESTO ES LO QUE FALTABA!)
        # samples es una lista de tuplas [('/path/img.jpg', label), ...]
        new_samples = []
        for path, old_label in ds.samples:
            new_samples.append((path, map_idx[old_label]))
        ds.samples = new_samples
        
        # Actualizamos el diccionario interno
        ds.class_to_idx = logica_medica

    # 3. Sampler (se mantiene igual, ahora usará los targets corregidos)
    targets = np.array(train_ds.targets)
    class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
    class_sample_count = np.where(class_sample_count == 0, 1, class_sample_count)
    weight = 1. / class_sample_count
    samples_weight = torch.from_numpy(np.array([weight[t] for t in targets]))
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # IMPORTANTE: Retornar las llaves de logica_medica para que el reporte salga en orden
    clases_ordenadas = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]
    
    return train_loader, val_loader, test_loader, clases_ordenadas
def get_class_weights(train_loader):
    # Esto sirve para penalizar más el error en clases pequeñas en la función de pérdida
    labels = [label for _, label in train_loader.dataset.samples]
    counter = Counter(labels)
    n_total = len(labels)
    k = len(counter)

    weights = torch.zeros(k)
    for cls_idx, count in counter.items():
        weights[cls_idx] = n_total / (k * count)
    return weights