import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

from conf import ID_TO_TRAINID



def get_output_shape(input_size: tuple, layer: nn.Module) -> tuple:
    """
    Restituisce la shape del tensore in uscita da un layer.
    """
    with torch.no_grad():
        x = torch.zeros(input_size)
        return tuple(layer(x).shape)


def get_mask_path(img_path: Path, gt_base: Path) -> Path:
    name = img_path.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
    return gt_base / img_path.parent.name / name


def encode_mask(target_mask_np: np.ndarray) -> np.ndarray:
    """Converte una maschera con ID grezzi (34 classi) in trainId (19 classi + 255)."""
    return ID_TO_TRAINID[target_mask_np]  


# Palette RGB ufficiale delle 19 classi trainId di Cityscapes
CITYSCAPES_PALETTE = np.array([
    [128, 64, 128],   # 0: road
    [244, 35, 232],   # 1: sidewalk
    [70, 70, 70],     # 2: building
    [102, 102, 156],  # 3: wall
    [190, 153, 153],  # 4: fence
    [153, 153, 153],  # 5: pole
    [250, 170, 30],   # 6: traffic light
    [220, 220, 0],    # 7: traffic sign
    [107, 142, 35],   # 8: vegetation
    [152, 251, 152],  # 9: terrain
    [70, 130, 180],   # 10: sky
    [220, 20, 60],    # 11: person
    [255, 0, 0],      # 12: rider
    [0, 0, 142],      # 13: car
    [0, 0, 70],       # 14: truck
    [0, 60, 100],     # 15: bus
    [0, 80, 100],     # 16: train
    [0, 0, 230],      # 17: motorcycle
    [119, 11, 32],    # 18: bicycle
], dtype=np.uint8)


def decode_segmap(mask_np: np.ndarray) -> np.ndarray:
    """Converte una maschera 2D con trainId (0..18, 255) in immagine RGB [H, W, 3]."""
    rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    valid_mask = mask_np != 255
    rgb[valid_mask] = CITYSCAPES_PALETTE[mask_np[valid_mask]]
    return rgb


def get_bilinear_kernel(in_channels: int, out_channels: int, kernel_size: int) -> torch.Tensor:
    """Genera un kernel 2D per inizializzare ConvTranspose2d come upsampling bilineare."""
    factor = (kernel_size + 1) // 2
    center = factor - 1 if kernel_size % 2 == 1 else factor - 0.5

    og = (
        torch.arange(kernel_size, dtype=torch.float32).reshape(-1, 1),
        torch.arange(kernel_size, dtype=torch.float32).reshape(1, -1),
    )
    filt = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)

    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=torch.float32)
    channels = min(in_channels, out_channels)
    weight[range(channels), range(channels), :, :] = filt
    return weight
