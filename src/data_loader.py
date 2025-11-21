import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from src.transforms import get_transform

cls_dict = {
    0: 'Disturbed',
    1: 'Merging',
    2: 'Round Smooth',
    3: 'Smooth, Cigar shaped',
    4: 'Cigar Shaped Smooth',
    5: 'Barred Spiral',
    6: 'Unbarred Tight Spiral',
    7: 'Unbarred Loose Spiral',
    8: 'Edge-on without Bulge',
    9: 'Edge-on with Bulge'
}

def cls_lookup(class_idx: int) -> str:
    """
    Obtener el nombre de la clase a partir del índice
    """
    return cls_dict.get(class_idx, "Unknown")

class Galaxy10Dataset(Dataset):
    """
    Wrapper dataset for Galaxy10 dataset from HuggingFace.
    """
    
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        image = item['image']
        label = item['label']
        
        if self.transform:
            image = self.transform(image)

        return image, label

def prepare_dataloaders(config):
    """
    Carga el dataset desde HuggingFace y prepara los dataloaders.
    
    Args:
        config: Objeto de configuración
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Cargar dataset desde HuggingFace
    print(f"Cargando dataset: {config.DATASET_NAME}")
    hf_dataset = load_dataset(config.DATASET_NAME)
    
    # El dataset de HF típicamente viene con 'train' y 'test' splits
    train_data = hf_dataset['train']
    test_data = hf_dataset['test']
    
    # Obtener transformaciones
    transform = get_transform(config)
    
    # Crear datasets
    train_dataset = Galaxy10Dataset(train_data, transform=transform)
    test_dataset = Galaxy10Dataset(test_data, transform=transform)
    
    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader