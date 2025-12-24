import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, Subset, DataLoader
import ssl


# Per evitare problemi di certificati SSL durante il download
ssl._create_default_https_context = ssl._create_unverified_context



# Trasformazioni base (resize e tensor)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Download EuroSAT
dataset = datasets.EuroSAT(
    root='data',  # cartella dove salvare
    transform=transform,
    download=True
)

print(f"Dataset dimensione: {len(dataset)} immagini")

# Suddivisione del dataset in train, val e test

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")


# Definizione delle trasformazioni con e senza augmentation

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),   # flip casuale
    transforms.RandomRotation(20),       # rotazione ±20°
    transforms.ToTensor(),               # converti in tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalizzazione
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Train dataset con augmentation
train_dataset.dataset.transform = train_transform

# Val e test senza augmentation
val_dataset.dataset.transform = val_transform
test_dataset.dataset.transform = val_transform

# Ora i dataset sono pronti per essere utilizzati nei DataLoader
batch_size = 32  # su Mac M2 va bene per la memoria

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train batch: {len(train_loader)}, Validation batch: {len(val_loader)}, Test batch: {len(test_loader)}")

