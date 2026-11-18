import os
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch.utils.data import DataLoader, Dataset

import conf
from transforms import get_train_transforms, get_val_transforms
from utils import encode_mask, get_mask_path


class CityscapeDataset(Dataset):
    def __init__(self, images: list[Path], masks: list[Path], transforms=None):
        super().__init__()
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        img = np.array(Image.open(self.images[index]).convert("RGB"))
        mask = np.array(Image.open(self.masks[index]))
        mask = encode_mask(mask)

        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask)
            img_aug = transformed["image"]
            mask_aug = transformed["mask"]
        else:
            img_aug = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask_aug = torch.from_numpy(mask)

        return img_aug, mask_aug

def _collect_and_verify_split(img_dir: Path, gt_dir: Path) -> tuple[list[Path], list[Path]]:
    images = sorted(img_dir.rglob("*_leftImg8bit.png"))
    masks = [get_mask_path(p, gt_dir) for p in images]
    missing = [m for m in masks if not m.exists()]
    assert not missing, f"Maschere mancanti in {gt_dir}: {len(missing)}"
    return images, masks


def build_dataloaders(
    root: Path = conf.DATA_ROOT,
    batch_size: int = conf.BATCH_SIZE,
    val_split: float = conf.VAL_SPLIT,
    seed: int = conf.SEED,
    height: int = conf.IMG_HEIGHT,
    width: int = conf.IMG_WIDTH,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    # 1. Raccolta percorsi train e test(val ufficiale di Cityscapes)
    all_train_imgs, all_train_masks = _collect_and_verify_split(
        root / "leftImg8bit" / "train",
        root / "gtFine" / "train",
    )
    test_imgs, test_masks = _collect_and_verify_split(
        root / "leftImg8bit" / "val",
        root / "gtFine" / "val",
    )

    # 2. Split Train / Val disgiunto per città
    groups = [p.parent.name for p in all_train_imgs]
    gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(gss.split(all_train_imgs, groups=groups))

    train_imgs = [all_train_imgs[i] for i in train_idx]
    train_masks = [all_train_masks[i] for i in train_idx]
    val_imgs = [all_train_imgs[i] for i in val_idx]
    val_masks = [all_train_masks[i] for i in val_idx]

    assert {p.parent.name for p in train_imgs}.isdisjoint({p.parent.name for p in val_imgs}), (
        "Data leakage: città condivise tra train e validation!"
    )

    # 3. trasformazioni
    train_tf = get_train_transforms(conf.IMAGENET_MEAN, conf.IMAGENET_STD, height=height, width=width)
    val_tf = get_val_transforms(conf.IMAGENET_MEAN, conf.IMAGENET_STD, height=height, width=width)

    # 4. Dataset e DataLoader
    train_ds = CityscapeDataset(train_imgs, train_masks, transforms=train_tf)
    val_ds = CityscapeDataset(val_imgs, val_masks, transforms=val_tf)
    test_ds = CityscapeDataset(test_imgs, test_masks, transforms=val_tf)

    num_workers = min(8, os.cpu_count() or 1)
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_dl = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_dl, val_dl, test_dl   