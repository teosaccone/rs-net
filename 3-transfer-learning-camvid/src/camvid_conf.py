"""Configurazione del punto 3 (transfer learning FCN Cityscapes -> CamVid).

Classi, palette e accorpamento 32 -> 11 presi dal notebook del collega
(2-deeplab-finetuning-camvid/differential-lr-finetuning.ipynb), così punto 2 e punto 3 usano le stesse etichette.

I moduli di questa cartella hanno il prefisso "camvid_" per non scontrarsi con conf.py, utils.py, engine.py
di 1-from-scratch-training/src (i test importano entrambe le cartelle). Reti FCN, metriche ed engine sono
COPIE di quelli del punto 1 (adattate a 11 classi): il codice comune si riunirà con un refactoring a fine progetto.
"""
import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Percorsi
# ---------------------------------------------------------------------------
# Risale da .../rs-net/3-transfer-learning-camvid/src/camvid_conf.py a .../rs-net
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Dataset Kaggle "carlolepelaars/camvid": cartelle train/, train_labels/, val/, val_labels/, test/, test_labels/
# (in locale in data/CamVid/; su Kaggle si imposta DATA_ROOT=/kaggle/input/.../CamVid)
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data" / "CamVid"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "outputs"))

# FCN addestrata su Cityscapes da cui si parte: la migliore del punto 1 (FCN-8s + CE + Dice, test mIoU 0.658)
CITYSCAPES_MODEL = "fcn8s"
CITYSCAPES_CKPT = OUTPUT_DIR / "fcn8s_ce_dice" / "best.pth"
# Classi Cityscapes nell'ordine dei trainId (copia di conf.CLASS_NAMES del punto 1; un test controlla che coincidano)
CITYSCAPES_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]

# ---------------------------------------------------------------------------
# CamVid: 11 classi semantiche + Void (ignorato)
# ---------------------------------------------------------------------------
N_CLASSES = 11
CLASS_NAMES = [
    "Sky", "Building", "Pole", "Road", "Sidewalk", "Tree",
    "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist",
]

# Colore con cui si disegna ogni classe nelle figure (palette ufficiale CamVid, ordine di CLASS_NAMES)
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

# Accorpamento delle 32 classi delle maschere (class_dict.csv) nelle 11 standard (raggruppamento di SegNet).
# Le classi "minori" non si scartano ma si accorpano alla classe madre (es. strisce -> Road, SUV -> Car).
# Void (0, 0, 0) non è nella tabella -> IGNORE_INDEX.
CAMVID_32_TO_11_RGB = {
    # 0: Sky
    (128, 128, 128): 0,  # Sky
    # 1: Building
    (128,   0,   0): 1,  # Building
    (  0, 128,  64): 1,  # Bridge
    ( 64, 192,   0): 1,  # Wall
    ( 64,   0,  64): 1,  # Tunnel
    (192,   0, 128): 1,  # Archway
    # 2: Pole
    (192, 192, 128): 2,  # Column_Pole
    (  0,   0,  64): 2,  # TrafficCone
    # 3: Road
    (128,  64, 128): 3,  # Road
    (128,   0, 192): 3,  # LaneMkgsDriv
    (192,   0,  64): 3,  # LaneMkgsNonDriv
    # 4: Sidewalk
    (  0,   0, 192): 4,  # Sidewalk
    ( 64, 192, 128): 4,  # ParkingBlock
    (128, 128, 192): 4,  # RoadShoulder
    # 5: Tree
    (128, 128,   0): 5,  # Tree
    (192, 192,   0): 5,  # VegetationMisc
    # 6: SignSymbol
    (192, 128, 128): 6,  # SignSymbol
    (128, 128,  64): 6,  # Misc_Text
    (  0,  64,  64): 6,  # TrafficLight
    # 7: Fence
    ( 64,  64, 128): 7,  # Fence
    # 8: Car
    ( 64,   0, 128): 8,  # Car
    ( 64, 128, 192): 8,  # SUVPickupTruck
    (192, 128, 192): 8,  # Truck_Bus
    (192,  64, 128): 8,  # Train
    (128,  64,  64): 8,  # OtherMoving
    # 9: Pedestrian
    ( 64,  64,   0): 9,  # Pedestrian
    (192, 128,  64): 9,  # Child
    ( 64,   0, 192): 9,  # CartLuggagePram
    ( 64, 128,  64): 9,  # Animal
    # 10: Bicyclist
    (  0, 128, 192): 10,  # Bicyclist
    (192,   0, 192): 10,  # MotorcycleScooter
}
IGNORE_INDEX = 255

# Classi Cityscapes (trainId 0..18, ordine di conf.CLASS_NAMES del punto 1) -> classe CamVid.
# Serve alla variante 1 (rete di Cityscapes valutata su CamVid senza addestramento): ogni predizione a
# 19 classi va convertita in una delle 11, nessuna può essere "ignorata".
CITYSCAPES_TO_CAMVID = {
    "road": "Road",
    "sidewalk": "Sidewalk",
    "building": "Building",
    "wall": "Building",          # in CamVid Wall è accorpato a Building
    "fence": "Fence",
    "pole": "Pole",
    "traffic light": "SignSymbol",  # in CamVid TrafficLight è accorpato a SignSymbol
    "traffic sign": "SignSymbol",
    "vegetation": "Tree",
    "terrain": "Tree",           # erba e terra: in CamVid VegetationMisc -> Tree (scelta nostra)
    "sky": "Sky",
    "person": "Pedestrian",
    "rider": "Bicyclist",        # in CamVid Bicyclist è persona + bici insieme
    "car": "Car",
    "truck": "Car",              # SUVPickupTruck, Truck_Bus -> Car
    "bus": "Car",
    "train": "Car",              # in CamVid Train -> Car
    "motorcycle": "Bicyclist",   # MotorcycleScooter -> Bicyclist
    "bicycle": "Bicyclist",
}

# ---------------------------------------------------------------------------
# Immagini
# ---------------------------------------------------------------------------
# Risoluzione scelta per tutto il punto 3: 720 x 960, quella nativa di CamVid (come il notebook del collega).
# Variante 1: risultato principale qui, più il confronto con 360x480 e 540x720 (EVAL_SIZES in zero_shot.py).
IMG_HEIGHT = 720
IMG_WIDTH = 960

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Training (varianti 2 e 3): li usano camvid_babysitting.py e camvid_train.py
# ---------------------------------------------------------------------------
MODEL = "fcn8s"        # fcn32s / fcn16s / fcn8s
LOSS = "ce_dice"       # ce / ce_dice (la migliore del punto 1)
# Da dove partono i pesi:
#   "imagenet"   variante 2: VGG16 ImageNet, testa nuova a 11 classi (come le run del punto 1, ma su CamVid)
#   "cityscapes" variante 3: tutta la FCN di Cityscapes (CITYSCAPES_CKPT), classificatori ridotti a 11 classi
INIT = "imagenet"
RUN_NAME = f"es3_{MODEL}_{LOSS}_{INIT}"  # cartella di output, es. outputs/es3_fcn8s_ce_dice_imagenet/

# Variante 3: per ogni classe CamVid, la classe Cityscapes "rappresentante" di cui si copiano i pesi dei
# classificatori (score_fr, score_pool4, score_pool3). Per le classi accorpate si sceglie la più frequente
# del gruppo (es. Car <- car, non truck/bus/train). Vedi camvid_engine.transfer_cityscapes_weights.
CITYSCAPES_REPRESENTATIVE = {
    "Sky": "sky",
    "Building": "building",       # + wall
    "Pole": "pole",
    "Road": "road",
    "Sidewalk": "sidewalk",
    "Tree": "vegetation",         # + terrain
    "SignSymbol": "traffic sign",  # + traffic light
    "Fence": "fence",
    "Car": "car",                 # + truck, bus, train
    "Pedestrian": "person",
    "Bicyclist": "bicycle",       # + rider, motorcycle
}

BATCH_SIZE = 4
SEED = 42
SCALE_RANGE = (0.75, 1.5)  # zoom casuale durante il training
