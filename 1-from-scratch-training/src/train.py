"""Training di FCN32s su Cityscapes (pensato per Kaggle).

Uso:
  python train.py              training completo
  DEBUG=1 python train.py      prova veloce: poche immagini, 1 epoca

Output in OUTPUT_DIR/train/: log.csv, curves.png, best.pth, test_metrics.json
"""
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # salva i grafici su file senza aprire finestre
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import conf
from dataset import build_dataloaders
from engine import build_model, evaluate, train_one_epoch

# ------------------------------------------------------------------ parametri
DEBUG = os.environ.get("DEBUG") == "1"
OUT_DIR = conf.OUTPUT_DIR / "train"

EPOCHS = 1 if DEBUG else 50
DEBUG_IMAGES = 80  # in DEBUG: 40 micro-batch = 10 passi dell'optimizer

# Batch effettivo 8 in entrambi i casi. Su Kaggle (16 GB) sta in un colpo solo;
# in locale (4 GB) si fanno 4 micro-batch da 2 e si accumulano i gradienti.
ON_KAGGLE = Path("/kaggle/working").exists()
BATCH_SIZE = 8 if ON_KAGGLE else 2   # immagini per micro-batch (quelle che stanno in VRAM)
ACCUM_STEPS = 1 if ON_KAGGLE else 4
EFFECTIVE_BATCH = BATCH_SIZE * ACCUM_STEPS

# Migliori valori della random search fine (outputs/babysitting/best_hparams.json, trovati con batch 4)
BEST_LR = 1.64e-3
BEST_WD = 6.65e-4
# Regola della radice: lr proporzionale a sqrt(batch). Più prudente della regola lineare
# (che con batch 8 darebbe 3.3e-3, già nella zona in cui la coarse peggiorava)
LR = BEST_LR * (EFFECTIVE_BATCH / 4) ** 0.5
WD = BEST_WD
# Gradient clipping: protezione dai picchi. Scelto misurando la norma SENZA clipping per ~1 epoca
# (batch effettivo 8, lr 2.3e-3): mediana 2.7, 99° percentile 6.6-7.5. Con 7 si taglia ~1% dei passi
# (solo i picchi, soprattutto all'inizio): la dinamica trovata col babysitting resta quasi intatta.
MAX_GRAD_NORM = 7.0


def save_history(history: list[dict]) -> None:
    """Riscrive log.csv e curves.png con tutte le epoche fatte finora."""
    with open(OUT_DIR / "log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_miou) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax_loss.set(xlabel="epoca", ylabel="loss")
    ax_loss.legend()
    ax_miou.plot(epochs, [h["val_miou"] for h in history])
    ax_miou.set(xlabel="epoca", ylabel="mIoU val")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "curves.png")
    plt.close(fig)


def summarize_grad_norms(norms: list[float]) -> dict[str, float]:
    """Mediana, 95° percentile e massimo della norma del gradiente, e % di passi tagliati.

    Serve a scegliere MAX_GRAD_NORM: deve stare un po' sopra i valori tipici (mediana),
    così il clipping scatta solo sui picchi. Se taglia quasi sempre, abbassa di fatto la lr.
    """
    finite = np.array([n for n in norms if np.isfinite(n)])  # i passi saltati dall'AMP danno inf/NaN
    return {
        "grad_norm_median": float(np.median(finite)),
        "grad_norm_p95": float(np.percentile(finite, 95)),
        "grad_norm_max": float(finite.max()),
        "clipped_pct": float((finite > MAX_GRAD_NORM).mean() * 100),
    }


def test_best_model(model, test_dl, criterion) -> None:
    """Valuta UNA volta sul test set il modello col miglior mIoU di validazione."""
    model.load_state_dict(torch.load(OUT_DIR / "best.pth"))
    test = evaluate(model, test_dl, criterion)
    print(f"\nTest: mIoU {test['miou']:.4f} | F1 {test['f1']:.4f} | pixel acc {test['pixel_acc']:.4f}")
    for name, iou in zip(conf.CLASS_NAMES, test["iou_per_class"]):
        print(f"    {name:15s} IoU {iou:.3f}")
    test["iou_per_class"] = dict(zip(conf.CLASS_NAMES, test["iou_per_class"]))
    (OUT_DIR / "test_metrics.json").write_text(json.dumps(test, indent=2))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Dati: {conf.DATA_ROOT} | output: {OUT_DIR} | DEBUG={DEBUG}")
    print(f"{EPOCHS} epoche, batch {BATCH_SIZE} x {ACCUM_STEPS} accumuli = {EFFECTIVE_BATCH}, "
          f"lr {LR:.2e}, wd {WD:.2e}")

    train_dl, val_dl, test_dl = build_dataloaders(BATCH_SIZE, hflip=True, limit=DEBUG_IMAGES if DEBUG else None)
    model, optimizer, scaler = build_model(LR, WD)
    # Poly decay per passo dell'optimizer: lr = LR * (1 - passo / passi_totali)^0.9, arriva a 0 alla fine.
    # Con l'accumulo i passi sono len(train_dl) // ACCUM_STEPS per epoca, non len(train_dl).
    total_iters = EPOCHS * (len(train_dl) // ACCUM_STEPS)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_iters, power=0.9)
    criterion = nn.CrossEntropyLoss(ignore_index=conf.IGNORE_INDEX)

    history, best_miou, ref_loss = [], -1.0, None
    for epoch in range(1, EPOCHS + 1):
        lr = scheduler.get_last_lr()[0]
        # ref_loss = loss iniziale della prima epoca: il controllo "esplosione" (> 3x) resta
        # riferito all'inizio, altrimenti un batch difficile a fine training lo farebbe scattare
        grad_norms = []
        losses, exploded = train_one_epoch(model, train_dl, optimizer, scaler, criterion, ref_loss=ref_loss,
                                           scheduler=scheduler, max_grad_norm=MAX_GRAD_NORM, grad_norms=grad_norms,
                                           accum_steps=ACCUM_STEPS)
        ref_loss = ref_loss or losses[0]
        if exploded:
            print("Loss esplosa: abbassa la LR.")
            break
        val = evaluate(model, val_dl, criterion)
        gn = summarize_grad_norms(grad_norms)

        history.append({"epoch": epoch, "lr": lr, "train_loss": float(np.mean(losses)),
                        "val_loss": val["loss"], "val_miou": val["miou"], **gn})
        save_history(history)
        print(f"epoca {epoch:2d}/{EPOCHS} | lr {lr:.2e} | train loss {history[-1]['train_loss']:.4f} "
              f"| val loss {val['loss']:.4f} | val mIoU {val['miou']:.4f}")
        print(f"    norma gradiente: mediana {gn['grad_norm_median']:.2f} | 95° perc. {gn['grad_norm_p95']:.2f} "
              f"| max {gn['grad_norm_max']:.2f} | tagliati {gn['clipped_pct']:.0f}% (soglia {MAX_GRAD_NORM})")

        if val["miou"] > best_miou:
            best_miou = val["miou"]
            torch.save(model.state_dict(), OUT_DIR / "best.pth")
            print("    -> nuovo miglior mIoU, salvato best.pth")

    test_best_model(model, test_dl, criterion)


if __name__ == "__main__":
    main()
