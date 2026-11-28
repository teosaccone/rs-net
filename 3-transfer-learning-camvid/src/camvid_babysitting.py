"""Babysitting ridotto su CamVid per la variante scelta in camvid_conf.INIT (copia adattata di babysitting.py).

Per i modelli nuovi niente random search completa (CLAUDE.md): rete e ricetta sono quelle del punto 1,
cambiano solo il dataset e i pesi di partenza. Fasi:
  1. loss iniziale: "imagenet" -> CE = ln(11) esatta (classificatori a zero, logit tutti 0);
                    "cityscapes" -> molto più bassa (la rete sa già segmentare)
  2. overfit su 20 immagini (la loss deve scendere molto)
  3. ricerca della lr (wd piccolo, poche epoche): quali lr fanno scendere la loss
  4. confronto rapido: COMPARE_TRIALS lr tra quelle buone, wd fisso, COMPARE_EPOCHS epoche, scelta per mIoU
     di validazione. Senza clipping: si misura la norma del gradiente per scegliere la soglia, e la VRAM.

Uso: python camvid_babysitting.py   (risultati in outputs/babysitting_<RUN_NAME>/)
Batch effettivo 8 (micro-batch 2 x accumulo 4), lo stesso di camvid_train.py: lr e wd si usano così come sono.
"""
import json
import math

import matplotlib

matplotlib.use("Agg")  # salva i grafici su file senza aprire finestre
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import camvid_conf as conf
from camvid_dataset import CamVidDataset, build_split, collect_split
from camvid_engine import build_model, dice_loss, evaluate, train_one_epoch
from camvid_transforms import get_val_transforms

# ------------------------------------------------------------------ parametri
OUT_DIR = conf.OUTPUT_DIR / f"babysitting_{conf.RUN_NAME}"
BATCH_SIZE = 2   # immagini per micro-batch (720x960 = 1.3 volte i pixel di 512x1024, dove 2 stavano in 4 GB: da misurare)
ACCUM_STEPS = 4  # 2 x 4 = batch effettivo 8, come camvid_train.py e il punto 1
DICE_WEIGHT = 1.0 if conf.LOSS == "ce_dice" else 0.0  # peso 1 scelto nel punto 1 (dice_weight_test.py)

OVERFIT_IMAGES = 20
OVERFIT_EPOCHS = 200
OVERFIT_LR = 1e-3

LR_CANDIDATES = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
LR_SEARCH_WD = 1e-6     # regolarizzazione piccola
LR_SEARCH_EPOCHS = 3    # 184 micro-batch per epoca -> ~550 micro-batch = ~138 aggiornamenti dei pesi
LR_MIN_DROP = 0.80      # "scende" = loss finale sotto l'80% di quella iniziale (come nel punto 1)

COMPARE_TRIALS = 4
COMPARE_EPOCHS = 10
COMPARE_WD = 6.65e-4    # wd della ricetta del punto 1 (run A)


# ------------------------------------------------------------------ fase 1
def check_initial_loss(batch) -> None:
    """CE e Dice iniziali; aggiungendo L2 la loss deve salire."""
    model, _, _ = build_model(lr=0.0, wd=0.0)
    model.eval()
    imgs, masks = batch
    with torch.no_grad():
        logits = model(imgs.cuda())
        masks = masks.cuda().long()
        ce = F.cross_entropy(logits, masks, ignore_index=conf.IGNORE_INDEX).item()
        dice = dice_loss(logits, masks).item()
    loss = ce + DICE_WEIGHT * dice
    l2 = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad).item()
    expected = (f"atteso ln({conf.N_CLASSES}) = {math.log(conf.N_CLASSES):.4f}" if conf.INIT == "imagenet"
                else f"deve essere ben sotto ln({conf.N_CLASSES}) = {math.log(conf.N_CLASSES):.4f}")
    print(f"\n[1] CE iniziale: {ce:.4f}  ({expected})")
    if DICE_WEIGHT > 0:
        print(f"    Dice iniziale: {dice:.4f} -> loss = CE + {DICE_WEIGHT:g} * Dice = {loss:.4f}")
    for lam in [1e-4, 1e-3]:
        print(f"    con L2 (lambda={lam:g}): {loss + lam * l2:.4f}  -> deve essere maggiore")


# ------------------------------------------------------------------ fase 2
def overfit_small_subset(batches) -> None:
    """Niente regolarizzazione (wd = 0, dropout spento): la loss deve scendere molto."""
    print(f"\n[2] Overfit su {OVERFIT_IMAGES} immagini")
    model, optimizer, scaler = build_model(OVERFIT_LR, wd=0.0, dropout=False)
    epoch_losses = []
    for epoch in range(1, OVERFIT_EPOCHS + 1):
        losses, exploded = train_one_epoch(model, batches, optimizer, scaler, DICE_WEIGHT)
        epoch_losses.append(np.mean(losses))
        if exploded:
            print("    loss esplosa: lr troppo alta")
            break
        if epoch % 20 == 0:
            m = evaluate(model, batches, DICE_WEIGHT)
            print(f"    epoca {epoch:3d} | loss {m['loss']:.4f} | pixel acc {m['pixel_acc']:.3f} | mIoU {m['miou']:.3f}")
    plt.figure()
    plt.plot(epoch_losses)
    plt.xlabel("epoca")
    plt.ylabel("train loss")
    plt.title(f"Overfit ({conf.INIT})")
    plt.savefig(OUT_DIR / "overfit.png")
    plt.close()


# ------------------------------------------------------------------ fase 3
def find_learning_rate(train_dl) -> list[float]:
    """Prova ogni lr per LR_SEARCH_EPOCHS epoche e restituisce quelle che fanno scendere la loss."""
    print(f"\n[3] Ricerca della lr (wd = {LR_SEARCH_WD:g}, {LR_SEARCH_EPOCHS} epoche)")
    good = []
    plt.figure()
    for lr in LR_CANDIDATES:
        model, optimizer, scaler = build_model(lr, LR_SEARCH_WD)
        train_dl.generator.manual_seed(conf.SEED)  # stesso ordine dei dati per ogni lr
        losses, exploded = [], False
        for _ in range(LR_SEARCH_EPOCHS):
            epoch_losses, exploded = train_one_epoch(model, train_dl, optimizer, scaler, DICE_WEIGHT,
                                                     ref_loss=losses[0] if losses else None, accum_steps=ACCUM_STEPS)
            losses += epoch_losses
            if exploded:
                break
        ratio = np.mean(losses[-30:]) / np.mean(losses[:10])
        if exploded:
            verdict = "esplode (troppo alta)"
        elif ratio > LR_MIN_DROP:
            verdict = f"quasi ferma (calo < {1 - LR_MIN_DROP:.0%}: troppo bassa)"
        else:
            verdict = "scende"
            good.append(lr)
        print(f"    lr {lr:.0e}: {verdict} | loss finale / iniziale {ratio:.2f}")
        plt.plot(losses, label=f"lr={lr:.0e}")
        del model, optimizer
        torch.cuda.empty_cache()
    plt.xlabel("iterazione")
    plt.ylabel("train loss")
    plt.legend()
    plt.savefig(OUT_DIR / "lr_search.png")
    plt.close()
    return good


# ------------------------------------------------------------------ fase 4
def compare_learning_rates(good_lrs: list[float], train_dl, val_dl) -> list[dict]:
    """COMPARE_TRIALS lr a distanza costante (in log) tra la più bassa e la più alta che scendono."""
    lrs = np.logspace(math.log10(min(good_lrs)), math.log10(max(good_lrs)), COMPARE_TRIALS)
    print(f"\n[4] Confronto di {COMPARE_TRIALS} lr (wd {COMPARE_WD:g}, {COMPARE_EPOCHS} epoche, senza clipping)")
    results = []
    for lr in lrs:
        model, optimizer, scaler = build_model(float(lr), COMPARE_WD)
        train_dl.generator.manual_seed(conf.SEED)
        torch.cuda.reset_peak_memory_stats()
        best_miou, ref_loss, grad_norms, exploded = 0.0, None, [], False
        for _ in range(COMPARE_EPOCHS):
            losses, exploded = train_one_epoch(model, train_dl, optimizer, scaler, DICE_WEIGHT, ref_loss=ref_loss,
                                               accum_steps=ACCUM_STEPS, grad_norms=grad_norms)
            ref_loss = ref_loss or losses[0]
            if exploded:
                break
            best_miou = max(best_miou, evaluate(model, val_dl, DICE_WEIGHT)["miou"])
        finite = np.array([n for n in grad_norms if np.isfinite(n)])  # i passi saltati dall'AMP danno inf
        r = {"lr": float(lr), "wd": COMPARE_WD, "miou": best_miou, "exploded": exploded,
             "grad_norm_median": float(np.median(finite)), "grad_norm_p99": float(np.percentile(finite, 99)),
             "vram_peak_mb": torch.cuda.max_memory_allocated() / 2 ** 20}
        results.append(r)
        print(f"    lr {lr:.2e} | mIoU val {best_miou:.4f} | norma gradiente mediana {r['grad_norm_median']:.2f}, "
              f"p99 {r['grad_norm_p99']:.2f} | VRAM {r['vram_peak_mb']:.0f} MB" + (" | esplosa" if exploded else ""))
        del model, optimizer
        torch.cuda.empty_cache()
    results.sort(key=lambda r: r["miou"], reverse=True)
    (OUT_DIR / "compare_lr.json").write_text(json.dumps(results, indent=2))
    return results


# ------------------------------------------------------------------ main
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Modello: {conf.MODEL} | loss: {conf.LOSS} | pesi di partenza: {conf.INIT} | "
          f"{conf.IMG_HEIGHT}x{conf.IMG_WIDTH} | batch {BATCH_SIZE} x {ACCUM_STEPS} accumuli | output: {OUT_DIR}")

    # Fasi 1-2: 20 immagini di train senza augmentation, caricate una volta in memoria
    images, masks = collect_split(conf.DATA_ROOT / "train", conf.DATA_ROOT / "train_labels")
    tiny = CamVidDataset(images[:OVERFIT_IMAGES], masks[:OVERFIT_IMAGES], get_val_transforms())
    imgs, labels = map(torch.stack, zip(*[tiny[i] for i in range(len(tiny))]))
    batches = [(imgs[i:i + BATCH_SIZE], labels[i:i + BATCH_SIZE]) for i in range(0, len(imgs), BATCH_SIZE)]
    check_initial_loss(batches[0])
    overfit_small_subset(batches)

    # Fasi 3-4: train con augmentation (come camvid_train.py), val senza
    train_dl = build_split("train", batch_size=BATCH_SIZE, augment=True)
    val_dl = build_split("val", batch_size=BATCH_SIZE)
    good_lrs = find_learning_rate(train_dl)
    if not good_lrs:
        print("Nessuna lr fa scendere la loss: controlla normalizzazione e mapping delle etichette.")
        return
    results = compare_learning_rates(good_lrs, train_dl, val_dl)
    best = results[0]
    print(f"\nMigliore: lr = {best['lr']:.2e}, wd = {best['wd']:.2e}, mIoU val = {best['miou']:.4f}; "
          f"soglia di clipping suggerita ~2.5 x mediana = {2.5 * best['grad_norm_median']:.1f} "
          f"(p99 {best['grad_norm_p99']:.1f})")
    (OUT_DIR / "best_hparams.json").write_text(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
