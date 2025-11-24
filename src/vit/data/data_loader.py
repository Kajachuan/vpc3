from datasets import load_dataset
from torch.utils.data import Dataset
import torch

class GalaxyDataset(Dataset):
    def __init__(self, hf_split, transform):
        self.hf_split = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        item = self.hf_split[idx]
        image = self.transform(item["image"])
        label = torch.tensor(item["label"], dtype=torch.long)
        return {"pixel_values": image, "labels": label}

def load_galaxy10(train_tf, test_tf):
    ds = load_dataset("matthieulel/galaxy10_decals")
    return (
        GalaxyDataset(ds["train"], train_tf),
        GalaxyDataset(ds["test"], test_tf)
    )
