import cv2
import torch
import torch.nn as nn
import numpy as np
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

def get_transform(config):
    """Obtener composición de transformaciones"""
    
    return transforms.Compose([
        MorphologicalOpeningTransform(config.MORPH_KERNEL_SIZE),
        transforms.RandomRotation(degrees=config.ROTATION_DEGREES),
        transforms.CenterCrop(size=(config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.RandomAffine(degrees=0, translate=config.TRANSLATE),
        transforms.ColorJitter(contrast=config.CONTRAST),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToDtype(torch.float32),
        transforms.Normalize(mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5)) # Normalizar a [-1, 1]
    ])