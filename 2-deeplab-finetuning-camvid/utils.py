import numpy as np
import torch

import conf


# ---------------------------------------------------------------------------
# Lookup table RGB → indice di classe
# ---------------------------------------------------------------------------
# Costruiamo una LUT 256×256×256 → uint8.
# Tutti i pixel non riconosciuti ottengono IGNORE_INDEX (255 = Void).
_RGB_TO_CLASS = np.full((256, 256, 256), conf.IGNORE_INDEX, dtype=np.uint8)
for class_idx, rgb in enumerate(conf.CAMVID_PALETTE):
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    _RGB_TO_CLASS[r, g, b] = class_idx


def encode_mask(mask_rgb: np.ndarray) -> np.ndarray:
    """Converte una maschera RGB [H, W, 3] in una maschera di indici [H, W].

    I pixel con colore non in palette diventano IGNORE_INDEX (255).
    """
    return _RGB_TO_CLASS[mask_rgb[..., 0], mask_rgb[..., 1], mask_rgb[..., 2]]


def decode_segmap(mask: np.ndarray) -> np.ndarray:
    """Converte una maschera di indici [H, W] in un'immagine RGB [H, W, 3]."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx, color in enumerate(conf.CAMVID_PALETTE):
        rgb[mask == class_idx] = color
    return rgb


# ---------------------------------------------------------------------------
# Utility generali
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    """Fissa i seed di random, numpy e torch per la riproducibilità."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
