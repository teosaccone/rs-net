import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    mean: tuple[float, ...],
    std: tuple[float, ...],
    height: int = 512,
    width: int = 1024
) -> A.Compose:
    return A.Compose([
        A.Resize(height=height, width=width),
        # Decommenta quando vuoi attivare l'augmentation:
        # A.HorizontalFlip(p=0.5),
        # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_transforms(
    mean: tuple[float, ...],
    std: tuple[float, ...],
    height: int = 512,
    width: int = 1024
) -> A.Compose:
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])