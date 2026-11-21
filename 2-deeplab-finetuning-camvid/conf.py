import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Percorsi
# ---------------------------------------------------------------------------
DATA_ROOT = Path(__file__).resolve().parent / "archive"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# ---------------------------------------------------------------------------
# Modello
# ---------------------------------------------------------------------------
MODEL = "deeplabv3"
LOSS  = "ce"

# ---------------------------------------------------------------------------
# CamVid: 11 classi semantiche + Void (ignorato)
# ---------------------------------------------------------------------------
N_CLASSES = 11

CLASS_NAMES = [
    "Sky",
    "Building",
    "Pole",
    "Road",
    "Sidewalk",
    "Tree",
    "SignSymbol",
    "Fence",
    "Car",
    "Pedestrian",
    "Bicyclist",
]

# Palette RGB ufficiale CamVid (una riga per classe, stesso ordine di CLASS_NAMES)
CAMVID_PALETTE = np.array([
    [128, 128, 128],   #  0 Sky
    [128,   0,   0],   #  1 Building
    [192, 192, 128],   #  2 Pole
    [128,  64, 128],   #  3 Road
    [  0,   0, 192],   #  4 Sidewalk
    [128, 128,   0],   #  5 Tree
    [192, 128, 128],   #  6 SignSymbol
    [ 64,  64, 128],   #  7 Fence
    [ 64,   0, 128],   #  8 Car
    [ 64,  64,   0],   #  9 Pedestrian
    [  0, 128, 192],   # 10 Bicyclist
], dtype=np.uint8)

# Pixel Void (0, 0, 0) → ignorato durante training e valutazione
IGNORE_INDEX = 255

# ---------------------------------------------------------------------------
# Immagini – CamVid nativo: 960 × 720;  si ridimensiona a 480 × 360
# ---------------------------------------------------------------------------
IMG_HEIGHT = 360
IMG_WIDTH  = 480

# Statistiche ImageNet (backbone ResNet pre-addestrato su ImageNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE  = 4
SEED        = 42
SCALE_RANGE = (0.75, 1.5)   # zoom casuale durante il training
