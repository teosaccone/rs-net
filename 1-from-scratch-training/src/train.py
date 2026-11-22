"""Training su Cityscapes della rete scelta in conf.MODEL (fcn32s o fcn16s).

Uso:
  python train.py              training completo
  DEBUG=1 python train.py      prova veloce: poche immagini, 1 epoca
  RESUME=1 python train.py     riprende un run interrotto da OUT_DIR/last.pth

Output in OUTPUT_DIR/<modello>_<loss>/ (es. fcn16s_ce/; con DEBUG fcn16s_ce_debug/):
log.csv, curves.png, best.pth, last.pth, test_metrics.json
(la run A della 32s, fatta prima di questa modifica, è in OUTPUT_DIR/train/)
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

import conf
from dataset import build_dataloaders
from engine import build_model, evaluate, train_one_epoch
from metrics import mean_iou

# ------------------------------------------------------------------ parametri
DEBUG = os.environ.get("DEBUG") == "1"
RESUME = os.environ.get("RESUME") == "1"  # esplicito: un run nuovo non riprende per sbaglio da uno vecchio
RUN_NAME = f"{conf.MODEL}_{conf.LOSS}"
OUT_DIR = conf.OUTPUT_DIR / (RUN_NAME + "_debug" if DEBUG else RUN_NAME)  # la prova non sovrascrive il run vero

EPOCHS = 1 if DEBUG else 80
DEBUG_IMAGES = 80  # in DEBUG: 40 micro-batch = 10 passi dell'optimizer

# Batch effettivo 8 in entrambi i casi. Su Kaggle (16 GB) sta in un colpo solo;
# in locale (4 GB) si fanno 4 micro-batch da 2 e si accumulano i gradienti.
ON_KAGGLE = Path("/kaggle/working").exists()
BATCH_SIZE = 8 if ON_KAGGLE else 2   # immagini per micro-batch (quelle che stanno in VRAM)
ACCUM_STEPS = 1 if ON_KAGGLE else 4
EFFECTIVE_BATCH = BATCH_SIZE * ACCUM_STEPS

# (lr, wd, dice_weight) per (modello, loss), batch effettivo 8, presi dal babysitting.
# dice_weight = peso lambda del Dice nella loss CE + lambda * Dice (0 per la sola CE).
# fcn32s: babysitting a batch 4 (outputs/babysitting/best_hparams.json: lr 1.64e-3, wd 6.65e-4),
#   lr portata a batch 8 con la regola della radice: lr proporzionale a sqrt(batch). Più prudente della
#   regola lineare (che darebbe 3.3e-3, già nella zona in cui la coarse peggiorava).
# fcn16s: stessa ricetta della 32s, per un confronto in cui cambia solo l'architettura.
#   Il babysitting della 16s (già a batch effettivo 8, outputs/babysitting_fcn16s/) lo conferma:
#   nella random search fine tutte le prove con lr 1.4e-3..5.9e-3 e wd 2e-4..2e-3 danno mIoU
#   0.518-0.526 (differenze = rumore); con lr 2.31e-3 mIoU 0.522, alla pari con la migliore
#   (lr 5.01e-3, wd 2.16e-4: 0.526, scartata perché cambierebbe la ricetta rispetto alla run A).
# fcn16s + ce_dice: stessa lr e wd della run B, peso Dice 1 (CE e Dice 1:1), così cambia solo la loss.
#   Babysitting (batch effettivo 8, outputs/babysitting_fcn16s_ce_dice/): nella fine le prove con
#   lr 3e-3..4e-3 e peso Dice 0.67..4.6 danno 0.537-0.544 (rumore); la migliore (lr 4.0e-3, wd 6.47e-4,
#   peso 0.67) ha wd quasi uguale alla run A. Lr più alte vincono in 5 epoche anche con la sola CE
#   (16s: 5.0e-3 migliore nella fine), quindi non è un motivo per cambiare la ricetta.
HPARAMS = {
    ("fcn32s", "ce"): (1.64e-3 * (8 / 4) ** 0.5, 6.65e-4, 0.0),
    ("fcn16s", "ce"): (1.64e-3 * (8 / 4) ** 0.5, 6.65e-4, 0.0),
    ("fcn16s", "ce_dice"): (1.64e-3 * (8 / 4) ** 0.5, 6.65e-4, 1.0),
}
LR, WD, DICE_WEIGHT = HPARAMS[(conf.MODEL, conf.LOSS)]
# Gradient clipping: protezione dai picchi. Scelto misurando la norma SENZA clipping per ~1 epoca
# (batch effettivo 8, lr 2.3e-3): mediana 2.7, 99° percentile 6.6-7.5. Con 7 si taglia ~1% dei passi
# (solo i picchi, soprattutto all'inizio): la dinamica trovata col babysitting resta quasi intatta.
# 16s + CE + Dice (misurato, 150 passi senza clipping): passi 1-50 mediana 3.7, p99 10 (sopra 7: 16%);
# passi 51-150 mediana 3.2, p99 6.0 (sopra 7: 1%). Stesso comportamento della CE: si tiene 7.
MAX_GRAD_NORM = 7.0


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


def save_checkpoint(state: dict) -> None:
    """Salva lo stato completo del run in last.pth.

    Prima scrive un file temporaneo e poi lo rinomina: se il PC si spegne proprio durante
    il salvataggio, il last.pth precedente resta intatto (la rinomina è istantanea).
    """
    tmp = OUT_DIR / "last.pth.tmp"
    torch.save(state, tmp)
    os.replace(tmp, OUT_DIR / "last.pth")


def test_best_model(model, test_dl) -> None:
    """Valuta UNA volta sul test set il modello col miglior mIoU di validazione."""
    model.load_state_dict(torch.load(OUT_DIR / "best.pth"))
    test = evaluate(model, test_dl, DICE_WEIGHT, desc="test")
    print(f"\nTest: mIoU {test['miou']:.4f} | F1 {test['f1']:.4f} | pixel acc {test['pixel_acc']:.4f}")
    for name, iou in zip(conf.CLASS_NAMES, test["iou_per_class"]):
        print(f"    {name:15s} IoU {iou:.3f}")
    test["iou_per_class"] = dict(zip(conf.CLASS_NAMES, test["iou_per_class"]))
    (OUT_DIR / "test_metrics.json").write_text(json.dumps(test, indent=2))


def main() -> None:
    assert LR is not None, f"Mancano gli iperparametri di {conf.MODEL} + {conf.LOSS} in HPARAMS: fai prima il babysitting"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Modello: {conf.MODEL} | loss: {conf.LOSS} | dati: {conf.DATA_ROOT} | output: {OUT_DIR} | DEBUG={DEBUG}")
    print(f"{EPOCHS} epoche, batch {BATCH_SIZE} x {ACCUM_STEPS} accumuli = {EFFECTIVE_BATCH}, "
          f"lr {LR:.2e}, wd {WD:.2e}, peso Dice {DICE_WEIGHT:g}")

    train_dl, val_dl, test_dl = build_dataloaders(BATCH_SIZE, augment=True, limit=DEBUG_IMAGES if DEBUG else None)
    model, optimizer, scaler = build_model(LR, WD)
    # Poly decay per passo dell'optimizer: lr = LR * (1 - passo / passi_totali)^0.9, arriva a 0 alla fine.
    # Con l'accumulo i passi sono len(train_dl) // ACCUM_STEPS per epoca, non len(train_dl).
    total_iters = EPOCHS * (len(train_dl) // ACCUM_STEPS)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_iters, power=0.9)

    history, best_miou, ref_loss, start_epoch = [], -1.0, None, 1
    if RESUME:
        # Per riprendere serve TUTTO lo stato, non solo i pesi: il momentum di SGD (optimizer),
        # il punto del poly decay (scheduler), la scala dell'AMP (scaler) e lo storico delle epoche.
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
        # ref_loss = loss iniziale della prima epoca: il controllo "esplosione" (> 3x) resta
        # riferito all'inizio, altrimenti un batch difficile a fine training lo farebbe scattare
        grad_norms = []
        # mIoU di train: calcolato sulle predizioni fatte durante il training stesso (gratis).
        # Attenzione: è su immagini aumentate, con il dropout attivo e con i pesi che cambiano
        # durante l'epoca, quindi è un po' pessimista rispetto al mIoU di val.
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
        print(f"epoca {epoch:2d}/{EPOCHS} | lr {lr:.2e} | train loss {history[-1]['train_loss']:.4f} "
              f"| val loss {val['loss']:.4f} | train mIoU {history[-1]['train_miou']:.4f} | val mIoU {val['miou']:.4f}")
        print(f"    norma gradiente: mediana {gn['grad_norm_median']:.2f} | 95° perc. {gn['grad_norm_p95']:.2f} "
              f"| max {gn['grad_norm_max']:.2f} | tagliati {gn['clipped_pct']:.0f}% (soglia {MAX_GRAD_NORM})")

        if val["miou"] > best_miou:
            best_miou = val["miou"]
            torch.save(model.state_dict(), OUT_DIR / "best.pth")
            print("    -> nuovo miglior mIoU, salvato best.pth")
        # Dopo best.pth, così un'interruzione tra i due salvataggi non perde il miglior modello
        save_checkpoint({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                         "history": history, "best_miou": best_miou, "ref_loss": ref_loss})

    if best_miou < 0:  # nessuna epoca completata (es. loss esplosa alla prima): niente da testare
        print("Nessun modello salvato in questo run: salto il test.")
        return
    test_best_model(model, test_dl)


if __name__ == "__main__":
    main()
