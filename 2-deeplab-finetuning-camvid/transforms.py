import albumentations as A
from albumentations.pytorch import ToTensorV2

import conf


def get_train_transforms(
    mean: tuple[float, ...] = conf.IMAGENET_MEAN,
    std:  tuple[float, ...] = conf.IMAGENET_STD,
    height: int = conf.IMG_HEIGHT,
    width:  int = conf.IMG_WIDTH,
    crop_size: tuple[int, int] | None = None,
    hflip: bool = False,
    scale_range: tuple[float, float] | None = None,
    color_jitter: bool = False,
) -> A.Compose:
    """Trasformazioni di train (tutte opzionali tranne resize e normalizzazione).

    Ordine fisso:
    1. Resize a height × width
    2. RandomScale  (se scale_range è fornito)
    3. RandomCrop   (se crop_size è fornito; padding con IGNORE_INDEX sulla maschera)
    4. HorizontalFlip (se hflip=True, p=0.5)
    5. ColorJitter  (se color_jitter=True, p=0.5)
    6. Normalize + ToTensorV2
    """
    ops = [A.Resize(height=height, width=width)]

    if scale_range is not None:
        ops.append(A.RandomScale(
            scale_limit=(scale_range[0] - 1, scale_range[1] - 1),
            p=1.0,
        ))

    if crop_size is not None:
        ops.append(A.RandomCrop(
            *crop_size,
            pad_if_needed=True,
            pad_position="random",
            fill=0,
            fill_mask=conf.IGNORE_INDEX,
        ))

    if hflip:
        ops.append(A.HorizontalFlip(p=0.5))

    if color_jitter:
        ops.append(A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.5,
        ))

    ops += [A.Normalize(mean=mean, std=std), ToTensorV2()]
    return A.Compose(ops)


def get_val_transforms(
    mean: tuple[float, ...] = conf.IMAGENET_MEAN,
    std:  tuple[float, ...] = conf.IMAGENET_STD,
    height: int = conf.IMG_HEIGHT,
    width:  int = conf.IMG_WIDTH,
) -> A.Compose:
    """Trasformazioni deterministiche per validation e test (nessuna augmentation)."""
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
