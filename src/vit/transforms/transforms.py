import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor
from torchvision.transforms import v2 as transforms

class MorphologicalOpeningTransform(nn.Module):
    """
    Transformación para aplicar apertura morfológica a imágenes
    """
    def __init__(self, kernel_size):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    def __call__(self, img):
        # PIL → numpy (H, W, 3)
        np_img = np.array(img)

        # aplicar por canal
        out = np.zeros_like(np_img)
        for c in range(3):
            out[:, :, c] = cv2.morphologyEx(np_img[:, :, c], cv2.MORPH_OPEN, self.kernel)

        # volver a PIL
        return Image.fromarray(out)

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
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

    test_tf = transforms.Compose([
        MorphologicalOpeningTransform(tuple(config["morph_kernel_size"])),
        transforms.CenterCrop(size=(config["img_height"], config["img_width"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])
    return train_tf, test_tf