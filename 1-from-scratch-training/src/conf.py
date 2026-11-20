from pathlib import Path
import numpy as np

# Risale da .../rs-net/1-from-scratch-training/src/conf.py a .../rs-net
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "Cityscapes"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

N_CLASSES = 19
INPUT_SIZE_CF = (1, 3, 512, 1024)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Parametri immagini e DataLoader
IMG_HEIGHT = 512
IMG_WIDTH = 1024
BATCH_SIZE = 8
VAL_SPLIT = 0.2
SEED = 42
IGNORE_INDEX = 255

# Crop di training (preso dopo il resize a IMG_HEIGHT x IMG_WIDTH): sta in 4 GB a batch 4
CROP_SIZE = (256, 512)

# Lookup table: indice = ID originale (0..33), valore = trainId (0..18) o 255
ID_TO_TRAINID = np.array([
    255, 255, 255, 255, 255, 255, 255, 0,   1, 255, 255, 2,   3,   4, 255, 255,
    255, 5,   255, 6,   7,   8,   9,   10,  11, 12,  13, 14,  15, 255, 255, 16,
    17,  18
], dtype=np.uint8)