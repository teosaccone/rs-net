import albumentations as A
from albumentations.pytorch import ToTensorV2

import conf


def get_train_transforms(
    mean: tuple[float, ...],
    std: tuple[float, ...],
    height: int = 512,
    width: int = 1024,
    crop_size: tuple[int, int] | None = None,
    hflip: bool = False,
    scale_range: tuple[float, float] | None = None,
    color_jitter: bool = False,
) -> A.Compose:
    """Trasformazioni di train. Ogni augmentation è opzionale e viene aggiunta in ordine alla lista.

    1. Resize a height x width (la scala "di riferimento", la stessa di val e test)
    2. scale_range=(a, b): zoom casuale di un fattore in [a, b] (immagine bilineare, maschera nearest)
    3. crop_size: ritaglio casuale; se l'immagine è più piccola (zoom < 1) si aggiunge padding,
       nero nell'immagine e IGNORE_INDEX nella maschera, così il padding non entra nella loss
    4. hflip: specchio orizzontale con p = 0.5
    5. color_jitter: luminosità, contrasto e saturazione ±20% con p = 0.5 (hue fissa: il colore
       di cielo e vegetazione è un indizio utile e non va alterato)
    """
    ops = [A.Resize(height=height, width=width)]
    if scale_range is not None:
        # RandomScale vuole lo scostamento da 1: (0.75, 1.5) -> (-0.25, +0.5)
        ops.append(A.RandomScale(scale_limit=(scale_range[0] - 1, scale_range[1] - 1), p=1.0))
    if crop_size is not None:
        ops.append(A.RandomCrop(*crop_size, pad_if_needed=True, pad_position="random",
                                fill=0, fill_mask=conf.IGNORE_INDEX))
    if hflip:
        ops.append(A.HorizontalFlip(p=0.5))
    if color_jitter:
        ops.append(A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.5))
    return A.Compose(ops + [A.Normalize(mean=mean, std=std), ToTensorV2()])


def get_val_transforms(
    mean: tuple[float, ...],
    std: tuple[float, ...],
    height: int = 512,
    width: int = 1024,
    crop_size: tuple[int, int] | None = None,
) -> A.Compose:
    # Deterministiche: niente augmentation, eventuale crop centrale
    ops = [A.Resize(height=height, width=width)]
    if crop_size is not None:
        ops.append(A.CenterCrop(*crop_size))
    return A.Compose(ops + [A.Normalize(mean=mean, std=std), ToTensorV2()])
