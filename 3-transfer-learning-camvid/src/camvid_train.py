"""Training su CamVid (varianti 2 e 3 del punto 3): copia adattata di train.py del punto 1.

Variante scelta in camvid_conf.INIT: "imagenet" (FCN da VGG16 ImageNet) o "cityscapes" (FCN di Cityscapes).
Stessa ricetta del punto 1: SGD momentum 0.9, poly 0.9 per passo, AMP, clipping, batch effettivo 8,
best.pth sul miglior mIoU di val, test valutato UNA volta con best.pth.

Uso:
  python camvid_train.py              training completo
  DEBUG=1 python camvid_train.py      prova veloce: poche immagini, 1 epoca
  RESUME=1 python camvid_train.py     riprende un run interrotto da OUT_DIR/last.pth

Output in OUTPUT_DIR/<RUN_NAME>/ (es. es3_fcn8s_ce_dice_imagenet/; con DEBUG ..._debug/):
log.csv, curves.png, best.pth, last.pth, test_metrics.json
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

import camvid_conf as conf
from camvid_dataset import build_dataloaders
from camvid_engine import build_model, evaluate, train_one_epoch
from camvid_metrics import mean_iou

# ------------------------------------------------------------------ parametri
DEBUG = os.environ.get("DEBUG") == "1"
RESUME = os.environ.get("RESUME") == "1"
RUN_NAME = conf.RUN_NAME
OUT_DIR = conf.OUTPUT_DIR / (RUN_NAME + "_debug" if DEBUG else RUN_NAME)

# Da decidere prima del lancio (il poly porta la lr a 0 all'ultima epoca), uguale per le due varianti.
# CamVid ha 369 immagini di train: 100 epoche = ~4600 aggiornamenti dei pesi (punto 1: 80 x 286 = ~22900).
EPOCHS = 1 if DEBUG else 100
DEBUG_IMAGES = 40  # in DEBUG: 20 micro-batch = 5 passi dell'optimizer

# Batch effettivo 8: su Kaggle in un colpo solo, in locale 4 micro-batch da 2 a 720x960 (come il punto 1
# a 512x1024; VRAM da misurare col babysitting)
ON_KAGGLE = Path("/kaggle/working").exists()
BATCH_SIZE = 8 if ON_KAGGLE else 2
ACCUM_STEPS = 1 if ON_KAGGLE else 4
EFFECTIVE_BATCH = BATCH_SIZE * ACCUM_STEPS

# Chiave = conf.RUN_NAME; valore = (lr, wd, peso Dice, soglia del gradient clipping).
# lr, wd e soglia: da camvid_babysitting.py (outputs/babysitting_<RUN_NAME>/best_hparams.json). None = da fare.
HPARAMS = {
    "es3_fcn8s_ce_dice_imagenet": (None, None, 1.0, None),
    "es3_fcn8s_ce_dice_cityscapes": (None, None, 1.0, None),
}
LR, WD, DICE_WEIGHT, MAX_GRAD_NORM = HPARAMS[RUN_NAME]


def save_history(history: list[dict]) -> None:
    """Riscrive log.csv e curves.png con tutte le epoche fatte finora."""
    with open(OUT_DIR / "log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_miou, ax_lr) = plt.subplots(1, 3, figsize=(15, 4))
    ax_loss.plot(epochs, [h["train_loss"] for h in history], marker=".", label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], marker=".", label="val")
    ax_loss.set(xlabel="epoca", ylabel=f"loss ({conf.LOSS})", title="Loss")
    ax_loss.legend()
    ax_miou.plot(epochs, [h["train_miou"] for h in history], marker=".", label="train")
    ax_miou.plot(epochs, [h["val_miou"] for h in history], marker=".", label="val")
    ax_miou.set(xlabel="epoca", ylabel="mIoU", title="mIoU")
    ax_miou.legend()
    ax_lr.plot(epochs, [h["lr"] for h in history], marker=".")
    ax_lr.set(xlabel="epoca", ylabel="lr", title="Learning rate (inizio epoca)")
    for ax in (ax_loss, ax_miou, ax_lr):
        ax.grid(alpha=0.3)
    fig.suptitle(RUN_NAME)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "curves.png")
    plt.close(fig)


def summarize_grad_norms(norms: list[float]) -> dict[str, float]:
    """Mediana, 95° percentile e massimo della norma del gradiente, e % di passi tagliati."""
    finite = np.array([n for n in norms if np.isfinite(n)])  # i passi saltati dall'AMP danno inf/NaN
    return {
        "grad_norm_median": float(np.median(finite)),
        "grad_norm_p95": float(np.percentile(finite, 95)),
        "grad_norm_max": float(finite.max()),
        "clipped_pct": float((finite > MAX_GRAD_NORM).mean() * 100),
    }


def save_checkpoint(state: dict) -> None:
    """Salva lo stato completo in last.pth (prima un file temporaneo, poi la rinomina istantanea)."""
    tmp = OUT_DIR / "last.pth.tmp"
    torch.save(state, tmp)
    os.replace(tmp, OUT_DIR / "last.pth")


def test_best_model(model, test_dl) -> None:
    """Valuta UNA volta sul test set il modello col miglior mIoU di validazione."""
    model.load_state_dict(torch.load(OUT_DIR / "best.pth"))
    test = evaluate(model, test_dl, DICE_WEIGHT, desc="test")
    print(f"\nTest: mIoU {test['miou']:.4f} | F1 {test['f1']:.4f} | pixel acc {test['pixel_acc']:.4f}")
    for name, iou in zip(conf.CLASS_NAMES, test["iou_per_class"]):
        print(f"    {name:12s} IoU {iou:.3f}")
    test["iou_per_class"] = dict(zip(conf.CLASS_NAMES, test["iou_per_class"]))
    (OUT_DIR / "test_metrics.json").write_text(json.dumps(test, indent=2))


def main() -> None:
    assert LR is not None and WD is not None and MAX_GRAD_NORM is not None, \
        f"Mancano gli iperparametri di {RUN_NAME} in HPARAMS: fai prima camvid_babysitting.py"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Modello: {conf.MODEL} | loss: {conf.LOSS} | pesi di partenza: {conf.INIT} | dati: {conf.DATA_ROOT} "
          f"| {conf.IMG_HEIGHT}x{conf.IMG_WIDTH} | output: {OUT_DIR} | DEBUG={DEBUG}")
    print(f"{EPOCHS} epoche, batch {BATCH_SIZE} x {ACCUM_STEPS} accumuli = {EFFECTIVE_BATCH}, "
          f"lr {LR:.2e}, wd {WD:.2e}, peso Dice {DICE_WEIGHT:g}, clipping {MAX_GRAD_NORM}")

    train_dl, val_dl, test_dl = build_dataloaders(batch_size=BATCH_SIZE, augment=True,
                                                  limit=DEBUG_IMAGES if DEBUG else None)
    model, optimizer, scaler = build_model(LR, WD)
    # Poly decay per passo dell'optimizer (con l'accumulo: len(train_dl) // ACCUM_STEPS passi per epoca)
    total_iters = EPOCHS * (len(train_dl) // ACCUM_STEPS)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_iters, power=0.9)

    history, best_miou, ref_loss, start_epoch = [], -1.0, None, 1
    if RESUME:
        # Tutto lo stato: pesi, momentum (optimizer), punto del poly (scheduler), scala AMP, storico
        ckpt = torch.load(OUT_DIR / "last.pth")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        history, best_miou, ref_loss = ckpt["history"], ckpt["best_miou"], ckpt["ref_loss"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Ripresa dall'epoca {start_epoch} (miglior mIoU finora {best_miou:.4f})")

    for epoch in range(start_epoch, EPOCHS + 1):
        lr = scheduler.get_last_lr()[0]
        grad_norms = []
        train_cm = torch.zeros((conf.N_CLASSES, conf.N_CLASSES), dtype=torch.long, device="cuda")
        losses, exploded = train_one_epoch(model, train_dl, optimizer, scaler, DICE_WEIGHT, ref_loss=ref_loss,
                                           scheduler=scheduler, max_grad_norm=MAX_GRAD_NORM, grad_norms=grad_norms,
                                           accum_steps=ACCUM_STEPS, cm=train_cm,
                                           desc=f"epoca {epoch}/{EPOCHS} train")
        ref_loss = ref_loss or losses[0]
        if exploded:
            print("Loss esplosa: abbassa la LR.")
            break
        val = evaluate(model, val_dl, DICE_WEIGHT, desc=f"epoca {epoch}/{EPOCHS} val")
        gn = summarize_grad_norms(grad_norms)

        history.append({"epoch": epoch, "lr": lr, "train_loss": float(np.mean(losses)),
                        "train_miou": mean_iou(train_cm), "val_loss": val["loss"], "val_miou": val["miou"], **gn})
        save_history(history)
        print(f"epoca {epoch:3d}/{EPOCHS} | lr {lr:.2e} | train loss {history[-1]['train_loss']:.4f} "
              f"| val loss {val['loss']:.4f} | train mIoU {history[-1]['train_miou']:.4f} | val mIoU {val['miou']:.4f}")
        print(f"    norma gradiente: mediana {gn['grad_norm_median']:.2f} | 95° perc. {gn['grad_norm_p95']:.2f} "
              f"| max {gn['grad_norm_max']:.2f} | tagliati {gn['clipped_pct']:.0f}% (soglia {MAX_GRAD_NORM})")

        if val["miou"] > best_miou:
            best_miou = val["miou"]
            torch.save(model.state_dict(), OUT_DIR / "best.pth")
            print("    -> nuovo miglior mIoU, salvato best.pth")
        save_checkpoint({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                         "history": history, "best_miou": best_miou, "ref_loss": ref_loss})

    if best_miou < 0:
        print("Nessun modello salvato in questo run: salto il test.")
        return
    test_best_model(model, test_dl)


if __name__ == "__main__":
    main()
