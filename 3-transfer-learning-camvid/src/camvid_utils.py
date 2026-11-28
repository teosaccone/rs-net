"""Conversione delle maschere CamVid e delle predizioni Cityscapes -> CamVid (punto 3), più seed e kernel
bilineare (copiati da utils.py del punto 1)."""
import random

import numpy as np
import torch

import camvid_conf as conf

# ---------------------------------------------------------------------------
# Maschere CamVid: colore RGB -> indice di classe (codice del collega)
# ---------------------------------------------------------------------------
# Lookup table 256x256x256 -> uint8: un accesso per pixel, senza cicli sulle classi.
# I colori non in tabella (Void e colori sconosciuti) diventano IGNORE_INDEX.
_RGB_TO_CLASS = np.full((256, 256, 256), conf.IGNORE_INDEX, dtype=np.uint8)
for (r, g, b), class_idx in conf.CAMVID_32_TO_11_RGB.items():
    _RGB_TO_CLASS[r, g, b] = class_idx


def encode_mask(mask_rgb: np.ndarray) -> np.ndarray:
    """Maschera RGB [H, W, 3] -> indici [H, W] (0..10 oppure IGNORE_INDEX)."""
    return _RGB_TO_CLASS[mask_rgb[..., 0], mask_rgb[..., 1], mask_rgb[..., 2]]


def decode_segmap(mask: np.ndarray) -> np.ndarray:
    """Indici [H, W] -> immagine RGB [H, W, 3]; i pixel ignorati restano neri."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx, color in enumerate(conf.CAMVID_PALETTE):
        rgb[mask == class_idx] = color
    return rgb


# ---------------------------------------------------------------------------
# Predizioni a 19 classi (Cityscapes) -> 11 classi (CamVid)
# ---------------------------------------------------------------------------
def cityscapes_to_camvid_index(cityscapes_names: list[str]) -> torch.Tensor:
    """Tensore [19]: per ogni trainId Cityscapes l'indice della classe CamVid corrispondente."""
    return torch.tensor([conf.CLASS_NAMES.index(conf.CITYSCAPES_TO_CAMVID[n]) for n in cityscapes_names])


def to_camvid_hard(logits: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    """Argmax sulle 19 classi, poi conversione: [B, 19, H, W] -> [B, H, W] con valori 0..10.

    Esempio: se vince "truck" il pixel diventa Car.
    """
    return mapping.to(logits.device)[logits.argmax(dim=1)]


def to_camvid_soft(logits: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    """Somma delle probabilità delle classi accorpate, poi argmax sulle 11: [B, 19, H, W] -> [B, H, W].

    Differenza con to_camvid_hard: se un pixel è car 0.30, truck 0.25, building 0.45, l'argmax a 19
    classi dà Building, ma sommando car + truck = 0.55 la classe CamVid Car vince. Le classi che in CamVid
    sono una sola (Car = car + truck + bus + train) non si "rubano" voti a vicenda.
    """
    probs = logits.float().softmax(dim=1)                              # [B, 19, H, W]
    groups = torch.nn.functional.one_hot(mapping, conf.N_CLASSES)     # [19, 11]: 1 se Cityscapes c -> CamVid k
    camvid_probs = torch.einsum("bchw,ck->bkhw", probs, groups.to(probs))  # somma per gruppo: [B, 11, H, W]
    return camvid_probs.argmax(dim=1)


# ---------------------------------------------------------------------------
# Copiati da utils.py del punto 1
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    """Fissa i seed di random, numpy e torch (CPU e GPU) per la riproducibilità."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_bilinear_kernel(in_channels: int, out_channels: int, kernel_size: int) -> torch.Tensor:
    """Kernel 2D per inizializzare ConvTranspose2d come upsampling bilineare (canale c -> canale c)."""
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
