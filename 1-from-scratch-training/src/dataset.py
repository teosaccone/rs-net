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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = np.array(Image.open(self.images[index]).convert("RGB"))
        mask = encode_mask(np.array(Image.open(self.masks[index])))

        if self.transforms is not None:
            out = self.transforms(image=img, mask=mask)
            return out["image"], out["mask"]
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0, torch.from_numpy(mask)


def _collect_and_verify_split(img_dir: Path, gt_dir: Path) -> tuple[list[Path], list[Path]]:
    images = sorted(img_dir.rglob("*_leftImg8bit.png"))
    masks = [get_mask_path(p, gt_dir) for p in images]
    missing = [m for m in masks if not m.exists()]
    assert not missing, f"Maschere mancanti in {gt_dir}: {len(missing)}"
    return images, masks


def split_train_val_test(root: Path = conf.DATA_ROOT, val_split: float = conf.VAL_SPLIT, seed: int = conf.SEED):
    """Restituisce (immagini, maschere) di train, val e test.

    Val = città del train ufficiale disgiunte dal nostro train; test = val ufficiale di Cityscapes.
    """
    imgs, masks = _collect_and_verify_split(root / "leftImg8bit" / "train", root / "gtFine" / "train")
    test = _collect_and_verify_split(root / "leftImg8bit" / "val", root / "gtFine" / "val")

    # Split disgiunto per città (evita leakage tra train e val)
    groups = [p.parent.name for p in imgs]
    gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(gss.split(imgs, groups=groups))
    train = ([imgs[i] for i in train_idx], [masks[i] for i in train_idx])
    val = ([imgs[i] for i in val_idx], [masks[i] for i in val_idx])

    assert {p.parent.name for p in train[0]}.isdisjoint({p.parent.name for p in val[0]}), (
        "Data leakage: città condivise tra train e validation!"
    )
    return train, val, test


def make_loader(ds: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    num_workers = min(8, os.cpu_count() or 1)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,  # nel train scartiamo l'ultimo batch incompleto
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(conf.SEED),  # ordine di shuffle riproducibile
    )


def build_dataloaders(
    batch_size: int = conf.BATCH_SIZE,
    height: int = conf.IMG_HEIGHT,
    width: int = conf.IMG_WIDTH,
    augment: bool = False,
    limit: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Loader di train, val e test. limit = numero massimo di immagini per split (per prove veloci).

    augment=True: zoom casuale + crop alla dimensione originale + flip + color jitter (solo train).
    """
    train, val, test = split_train_val_test()
    if limit is not None:
        train, val, test = [(imgs[:limit], masks[:limit]) for imgs, masks in (train, val, test)]
    if augment:
        train_tf = get_train_transforms(conf.IMAGENET_MEAN, conf.IMAGENET_STD, height, width,
                                        crop_size=(height, width), hflip=True,
                                        scale_range=conf.SCALE_RANGE, color_jitter=True)
    else:
        train_tf = get_train_transforms(conf.IMAGENET_MEAN, conf.IMAGENET_STD, height, width)
    val_tf = get_val_transforms(conf.IMAGENET_MEAN, conf.IMAGENET_STD, height, width)
    return (
        make_loader(CityscapeDataset(*train, transforms=train_tf), batch_size, shuffle=True),
        make_loader(CityscapeDataset(*val, transforms=val_tf), batch_size, shuffle=False),
        make_loader(CityscapeDataset(*test, transforms=val_tf), batch_size, shuffle=False),
    )
