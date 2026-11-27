"""Test del peso lambda del Dice (loss = CE + lambda * Dice) a lr e wd fissi.

Nella random search del babysitting lambda cambiava insieme a lr e wd, quindi il suo effetto non si
poteva separare (tutte le prove migliori tra 0.537 e 0.544). Qui lr e wd sono quelli della run A e
cambia SOLO lambda. lambda = 0 è la sola CE (stesso setup, fa da riferimento).

Oltre al mIoU si guardano le classi piccole: il Dice dovrebbe alzarne la recall e il rapporto
pixel predetti / pixel veri (con la sola CE sono sotto-predette, vedi appunti_relazione.txt, 4c).

Stesso setup delle prove "fine" di babysitting.py: crop 256x512 + flip, batch 4 x accumulo 2,
5 epoche, val a 512x1024; per ogni lambda si tiene l'epoca con il mIoU di val migliore.
Uso: python dice_weight_test.py   (risultati in outputs/dice_weight_test_<modello>/)
"""
import json

import numpy as np
import torch

import conf
from babysitting import ACCUM_STEPS, BATCH_SIZE
from dataset import CityscapeDataset, make_loader, split_train_val_test
from engine import build_model, evaluate, train_one_epoch
from transforms import get_train_transforms, get_val_transforms

# ------------------------------------------------------------------ parametri
OUT_DIR = conf.OUTPUT_DIR / f"dice_weight_test_{conf.MODEL}"
LR = 1.64e-3 * (8 / 4) ** 0.5  # ricetta della run A (come in train.py)
WD = 6.65e-4
DICE_WEIGHTS = [0.0, 0.5, 1.0, 2.0, 4.0]
EPOCHS = 5
# Classi piccole da controllare (sotto-predette con la sola CE)
SMALL_CLASSES = ["wall", "fence", "pole", "traffic light", "traffic sign", "rider", "motorcycle"]


def class_stats(cm: np.ndarray) -> dict[str, dict[str, float]]:
    """Per ogni classe: IoU, recall e rapporto pixel predetti / pixel veri (cm[vera][predetta])."""
    tp, gt, pred = np.diag(cm), cm.sum(1), cm.sum(0)
    return {name: {"iou": tp[i] / (gt[i] + pred[i] - tp[i]), "recall": tp[i] / gt[i], "pred_gt": pred[i] / gt[i]}
            for i, name in enumerate(conf.CLASS_NAMES)}


def run_one(dice_weight: float, train_dl, val_dl) -> dict:
    """Allena EPOCHS epoche con il peso dato e restituisce le metriche dell'epoca migliore."""
    model, optimizer, scaler = build_model(LR, WD)
    train_dl.generator.manual_seed(conf.SEED)  # stesso ordine dei dati per ogni peso
    best, ref_loss = None, None
    for epoch in range(1, EPOCHS + 1):
        losses, exploded = train_one_epoch(model, train_dl, optimizer, scaler, dice_weight, ref_loss=ref_loss,
                                           accum_steps=ACCUM_STEPS)
        ref_loss = ref_loss or losses[0]
        if exploded:
            print(f"    peso {dice_weight:g}: loss esplosa all'epoca {epoch}")
            break
        m = evaluate(model, val_dl, dice_weight)
        print(f"    peso {dice_weight:g} | epoca {epoch} | mIoU {m['miou']:.4f}")
        if best is None or m["miou"] > best["miou"]:
            best = {"epoch": epoch, "miou": m["miou"], "f1": m["f1"], "pixel_acc": m["pixel_acc"],
                    "classes": class_stats(np.array(m["confusion_matrix"], dtype=float))}
    del model, optimizer
    torch.cuda.empty_cache()
    return {"dice_weight": dice_weight, **best}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Modello: {conf.MODEL} | lr {LR:.2e} | wd {WD:.2e} | pesi Dice {DICE_WEIGHTS} | {EPOCHS} epoche")
    mean, std = conf.IMAGENET_MEAN, conf.IMAGENET_STD
    train, val, _ = split_train_val_test()
    train_dl = make_loader(CityscapeDataset(*train, get_train_transforms(mean, std, crop_size=conf.CROP_SIZE, hflip=True)),
                           BATCH_SIZE, shuffle=True)
    val_dl = make_loader(CityscapeDataset(*val, get_val_transforms(mean, std)), 2, shuffle=False)

    results = []
    for w in DICE_WEIGHTS:
        results.append(run_one(w, train_dl, val_dl))
        # Salva dopo ogni peso: se si interrompe, i risultati già fatti restano
        (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))

    # Tabella riassuntiva: mIoU e, per le classi piccole, IoU / recall / pred_gt
    print(f"\n{'peso':>5} {'ep':>3} {'mIoU':>7} | " + " | ".join(f"{c[:12]:>18}" for c in SMALL_CLASSES))
    print(f"{'':>17} | " + " | ".join(f"{'IoU rec pr/gt':>18}" for _ in SMALL_CLASSES))
    for r in results:
        cells = [f"{r['classes'][c]['iou']:.2f} {r['classes'][c]['recall']:.2f} {r['classes'][c]['pred_gt']:.2f}"
                 for c in SMALL_CLASSES]
        print(f"{r['dice_weight']:5g} {r['epoch']:3d} {r['miou']:7.4f} | " + " | ".join(f"{x:>18}" for x in cells))


if __name__ == "__main__":
    main()
