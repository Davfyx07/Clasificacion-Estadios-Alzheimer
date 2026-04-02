import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

# Configuración de Normalización estándar (ImageNet)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# TRANSFORMACIÓN DE ENTRENAMIENTO (Dinámica)
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    # Eliminamos Flip horizontal para no dañar la asimetría médica
    transforms.RandomRotation(5), # Rotación mínima
    transforms.ColorJitter(brightness=0.05, contrast=0.05), # Cambios de brillo casi imperceptibles
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# VALIDACIÓN Y TEST (Estáticas y Puras)
transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

def get_dataloaders(data_dir, batch_size=32):
    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=transform_train)
    val_ds   = datasets.ImageFolder(f"{data_dir}/val", transform=transform_eval)
    test_ds  = datasets.ImageFolder(f"{data_dir}/test", transform=transform_eval)

    # SAMPLER: Este es el motor que balancea sin necesidad de tener millones de fotos
    targets = train_ds.targets
    class_count = torch.tensor([(torch.tensor(targets) == t).sum() for t in range(len(train_ds.classes))])
    weight = 1. / class_count.float()
    samples_weight = torch.tensor([weight[t] for t in targets])
    
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader