"""Dataset e DataLoader di CamVid (codice del collega, dal notebook del punto 2).

Split ufficiale già separato nelle cartelle del dataset Kaggle: train (367), val (101), test (233).
Maschera di un'immagine: stesso nome con suffisso _L (0001TP_009210.png -> 0001TP_009210_L.png).
"""
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import camvid_conf as conf
from camvid_transforms import get_train_transforms, get_val_transforms
from camvid_utils import encode_mask


class CamVidDataset(Dataset):
    """Restituisce (immagine [3, H, W] float32 normalizzata, maschera [H, W] int64 con 0..10 o 255)."""

    def __init__(self, images: list[Path], masks: list[Path], transforms=None):
        super().__init__()
        assert len(images) == len(masks), "Il numero di immagini e maschere deve coincidere."
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = np.array(Image.open(self.images[index]).convert("RGB"))
        mask = encode_mask(np.array(Image.open(self.masks[index]).convert("RGB")))
        out = self.transforms(image=img, mask=mask)
        return out["image"], out["mask"].long()


def collect_split(img_dir: Path, mask_dir: Path) -> tuple[list[Path], list[Path]]:
    """Immagini PNG di img_dir e maschere corrispondenti (suffisso _L) in mask_dir."""
    images = sorted(img_dir.glob("*.png"))
    masks = [mask_dir / (img.stem + "_L.png") for img in images]
    missing = [m for m in masks if not m.exists()]
    assert images, f"Nessuna immagine in {img_dir}: scarica CamVid in {conf.DATA_ROOT} (o imposta DATA_ROOT)"
    assert not missing, f"Maschere mancanti in {mask_dir}: {missing[:3]}"
    return images, masks


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    """DataLoader standard: drop_last solo in training (niente batch da 1), worker persistenti."""
    num_workers = min(4, os.cpu_count() or 1)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(conf.SEED),
    )


def build_split(split: str, height: int = conf.IMG_HEIGHT, width: int = conf.IMG_WIDTH,
                batch_size: int = conf.BATCH_SIZE, augment: bool = False) -> DataLoader:
    """Loader di un solo split ("train", "val" o "test"); augment vale solo per "train"."""
    images, masks = collect_split(conf.DATA_ROOT / split, conf.DATA_ROOT / f"{split}_labels")
    if split == "train" and augment:
        tf = get_train_transforms(height=height, width=width, crop_size=(height, width), hflip=True,
                                  scale_range=conf.SCALE_RANGE, color_jitter=True)
    else:
        tf = get_val_transforms(height=height, width=width)
    return make_loader(CamVidDataset(images, masks, tf), batch_size, shuffle=split == "train")


def build_dataloaders(height: int = conf.IMG_HEIGHT, width: int = conf.IMG_WIDTH,
                      batch_size: int = conf.BATCH_SIZE,
                      augment: bool = False) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Loader di train, val e test (augmentation solo sul train, se richiesta)."""
    return tuple(build_split(s, height, width, batch_size, augment) for s in ("train", "val", "test"))
