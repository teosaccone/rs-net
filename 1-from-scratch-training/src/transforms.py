import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    mean: tuple[float, ...],
    std: tuple[float, ...],
    height: int = 512,
    width: int = 1024,
    crop_size: tuple[int, int] | None = None,
    hflip: bool = False,
) -> A.Compose:
    # Il crop si fa dopo il resize: gli oggetti hanno la scala di 512x1024, la VRAM quella del crop
    ops = [A.Resize(height=height, width=width)]
    if crop_size is not None:
        ops.append(A.RandomCrop(*crop_size))
    if hflip:
        ops.append(A.HorizontalFlip(p=0.5))
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
