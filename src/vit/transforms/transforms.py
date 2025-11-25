import cv2
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoImageProcessor
from torchvision.transforms import v2 as transforms

class MorphologicalOpeningTransform(nn.Module):
    """
    Transformación para aplicar apertura morfológica a imágenes
    """
    
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    
    def forward(self, img):
        # Asegurar que es numpy
        np_img = img if isinstance(img, np.ndarray) else img.numpy()
        
        # Aplicar morfología a cada canal
        morphs = []
        for ch in range(3):
            morphs.append(
                torch.from_numpy(
                    cv2.morphologyEx(np_img[ch], cv2.MORPH_OPEN, self.kernel)
                ).float()
            )
        
        return torch.stack(morphs)

def get_transforms(config: dict):
    """Obtener composición de transformaciones"""

    processor = AutoImageProcessor.from_pretrained(config["checkpoint"])
    
    train_tf = transforms.Compose([
        MorphologicalOpeningTransform(tuple(config["morph_kernel_size"])),
        transforms.RandomRotation(degrees=config["rotation_degrees"]),
        transforms.CenterCrop(size=(config["img_height"], config["img_width"])),
        transforms.RandomAffine(degrees=0, translate=tuple(config["translate"])),
        transforms.ColorJitter(contrast=config["contrast"]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToDtype(torch.float32),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

    test_tf = transforms.Compose([
        MorphologicalOpeningTransform(tuple(config["morph_kernel_size"])),
        transforms.CenterCrop(size=(config["img_height"], config["img_width"])),
        transforms.ToDtype(torch.float32),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])
    return train_tf, test_tf