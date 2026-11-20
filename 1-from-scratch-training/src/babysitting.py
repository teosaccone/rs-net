"""Babysitting del training di FCN32s (slide cs231n "Training Neural Networks part 2").

Esegue in ordine:
  1. check della loss iniziale
  2. overfit su 20 immagini
  3. ricerca della lr con regolarizzazione piccola
  4. random search coarse, poi fine sull'intervallo ristretto

Uso: python babysitting.py   (i risultati vanno in outputs/babysitting/)
"""
import json
import math

import matplotlib

matplotlib.use("Agg")  # salva i grafici su file senza aprire finestre
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import conf
from dataset import CityscapeDataset, make_loader, split_train_val_test
from engine import build_model, evaluate, train_one_epoch
from transforms import get_train_transforms, get_val_transforms

# ------------------------------------------------------------------ parametri
OUT_DIR = conf.OUTPUT_DIR / "babysitting"
BATCH_SIZE = 4

OVERFIT_IMAGES = 20
OVERFIT_EPOCHS = 200
OVERFIT_LR = 1e-3

LR_CANDIDATES = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
LR_SEARCH_WD = 1e-6   # regolarizzazione piccola
LR_SEARCH_STEPS = 300

WD_EXP = (-6, -2)     # weight decay = 10^U(-6, -2) nella fase coarse
COARSE_TRIALS, COARSE_EPOCHS = 12, 2
FINE_TRIALS, FINE_EPOCHS = 6, 5


# ------------------------------------------------------------------ fase 1
def check_initial_loss(batch, criterion) -> None:
    """Senza regolarizzazione loss ~ ln(19); aggiungendo L2 la loss deve salire."""
    model, _, _ = build_model(lr=0.0, wd=0.0)
    model.eval()  # niente dropout
    imgs, masks = batch
    with torch.no_grad():
        loss = criterion(model(imgs.cuda()), masks.cuda().long()).item()
    l2 = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad).item()

    print(f"\n[1] Loss iniziale: {loss:.4f}  (atteso ln({conf.N_CLASSES}) = {math.log(conf.N_CLASSES):.4f})")
    for lam in [1e-4, 1e-3]:
        print(f"    con L2 (lambda={lam:g}): {loss + lam * l2:.4f}  -> deve essere maggiore")


# ------------------------------------------------------------------ fase 2
def overfit_small_subset(batches, criterion) -> None:
    """Niente regolarizzazione (wd = 0, dropout spento): la loss deve scendere molto.

    Non arriva a 0: FCN32s ricostruisce l'output da una mappa a stride 32, i dettagli fini si perdono.
    """
    print(f"\n[2] Overfit su {OVERFIT_IMAGES} immagini")
    model, optimizer, scaler = build_model(OVERFIT_LR, wd=0.0, dropout=False)
    epoch_losses = []
    for epoch in range(1, OVERFIT_EPOCHS + 1):
        losses, exploded = train_one_epoch(model, batches, optimizer, scaler, criterion)
        epoch_losses.append(np.mean(losses))
        if exploded:
            print("    loss esplosa: lr troppo alta")
            break
        if epoch % 20 == 0:
            m = evaluate(model, batches, criterion)  # valutazione sulle stesse immagini
            print(f"    epoca {epoch:3d} | loss {m['loss']:.4f} | pixel acc {m['pixel_acc']:.3f} | mIoU {m['miou']:.3f}")

    plt.figure()
    plt.plot(epoch_losses)
    plt.xlabel("epoca")
    plt.ylabel("train loss")
    plt.title("Overfit")
    plt.savefig(OUT_DIR / "overfit.png")
    plt.close()


# ------------------------------------------------------------------ fase 3
def find_learning_rate(train_dl, criterion) -> list[float]:
    """Prova ogni lr per qualche centinaio di passi e restituisce quelle che fanno scendere la loss."""
    print(f"\n[3] Ricerca della lr (wd = {LR_SEARCH_WD:g}, {LR_SEARCH_STEPS} passi)")
    good = []
    plt.figure()
    for lr in LR_CANDIDATES:
        model, optimizer, scaler = build_model(lr, LR_SEARCH_WD)
        train_dl.generator.manual_seed(conf.SEED)  # stesso ordine dei dati per ogni lr
        losses, exploded = train_one_epoch(model, train_dl, optimizer, scaler, criterion, max_steps=LR_SEARCH_STEPS)

        if exploded:
            verdict = "esplode (troppo alta)"
        elif np.mean(losses[-30:]) > 0.95 * np.mean(losses[:10]):
            verdict = "quasi ferma (troppo bassa)"
        else:
            verdict = "scende"
            good.append(lr)
        print(f"    lr {lr:.0e}: {verdict}")
        plt.plot(losses, label=f"lr={lr:.0e}")

    plt.xlabel("iterazione")
    plt.ylabel("train loss")
    plt.legend()
    plt.savefig(OUT_DIR / "lr_search.png")
    plt.close()
    return good


# ------------------------------------------------------------------ fase 4
def random_search(name, lr_exp, wd_exp, trials, epochs, train_dl, val_dl, criterion) -> list[dict]:
    """Campiona lr e wd in scala logaritmica, allena per poche epoche, ordina per mIoU di validazione."""
    print(f"\n[4] Random search {name}: lr = 10^U({lr_exp[0]:.1f}, {lr_exp[1]:.1f}), "
          f"wd = 10^U({wd_exp[0]:.1f}, {wd_exp[1]:.1f}), {trials} prove x {epochs} epoche")
    rng = np.random.default_rng(conf.SEED)
    results = []
    for i in range(trials):
        lr = 10 ** rng.uniform(*lr_exp)
        wd = 10 ** rng.uniform(*wd_exp)
        model, optimizer, scaler = build_model(lr, wd)
        train_dl.generator.manual_seed(conf.SEED)

        best_miou, ref_loss = 0.0, None
        for _ in range(epochs):
            losses, exploded = train_one_epoch(model, train_dl, optimizer, scaler, criterion, ref_loss=ref_loss)
            ref_loss = ref_loss or losses[0]
            if exploded:
                break
            best_miou = max(best_miou, evaluate(model, val_dl, criterion)["miou"])

        results.append({"lr": lr, "wd": wd, "miou": best_miou, "exploded": exploded})
        print(f"    {i + 1:2d}/{trials} | lr {lr:.2e} | wd {wd:.2e} | mIoU {best_miou:.4f}" + (" | esplosa" if exploded else ""))
        del model, optimizer
        torch.cuda.empty_cache()

    results.sort(key=lambda r: r["miou"], reverse=True)
    (OUT_DIR / f"random_search_{name}.json").write_text(json.dumps(results, indent=2))

    # Slide 11: se il migliore è sul bordo dell'intervallo, l'intervallo va spostato
    best = results[0]
    for key, (lo, hi) in [("lr", lr_exp), ("wd", wd_exp)]:
        x = math.log10(best[key])
        if min(x - lo, hi - x) < 0.1 * (hi - lo):
            print(f"    ATTENZIONE: il miglior {key} è vicino al bordo dell'intervallo, conviene allargarlo")
    return results


def narrow_range(results: list[dict], key: str, top_k: int = 3) -> tuple[float, float]:
    """Nuovo intervallo (in log10) attorno ai top_k migliori, con un piccolo margine."""
    logs = [math.log10(r[key]) for r in results[:top_k]]
    return min(logs) - 0.25, max(logs) + 0.25


# ------------------------------------------------------------------ main
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss(ignore_index=conf.IGNORE_INDEX)
    mean, std = conf.IMAGENET_MEAN, conf.IMAGENET_STD
    train, val, _ = split_train_val_test()

    # Fasi 1-2: poche immagini, crop centrale fisso (niente augmentation), caricate una volta in memoria
    tiny = CityscapeDataset(train[0][:OVERFIT_IMAGES], train[1][:OVERFIT_IMAGES],
                            get_val_transforms(mean, std, crop_size=conf.CROP_SIZE))
    imgs, masks = map(torch.stack, zip(*[tiny[i] for i in range(len(tiny))]))
    batches = [(imgs[i:i + BATCH_SIZE], masks[i:i + BATCH_SIZE]) for i in range(0, len(imgs), BATCH_SIZE)]

    check_initial_loss(batches[0], criterion)
    overfit_small_subset(batches, criterion)

    # Fasi 3-4: train con random crop + flip, val sull'immagine intera 512x1024
    train_dl = make_loader(CityscapeDataset(*train, get_train_transforms(mean, std, crop_size=conf.CROP_SIZE, hflip=True)),
                           BATCH_SIZE, shuffle=True)
    val_dl = make_loader(CityscapeDataset(*val, get_val_transforms(mean, std)), 2, shuffle=False)

    good_lrs = find_learning_rate(train_dl, criterion)
    if not good_lrs:
        print("Nessuna lr fa scendere la loss: controlla normalizzazione e mapping delle etichette.")
        return
    # Intervallo coarse: le lr che scendono, più mezza decade per lato
    lr_exp = (math.log10(min(good_lrs)) - 0.5, math.log10(max(good_lrs)) + 0.5)

    coarse = random_search("coarse", lr_exp, WD_EXP, COARSE_TRIALS, COARSE_EPOCHS, train_dl, val_dl, criterion)
    fine = random_search("fine", narrow_range(coarse, "lr"), narrow_range(coarse, "wd"),
                         FINE_TRIALS, FINE_EPOCHS, train_dl, val_dl, criterion)

    best = fine[0]
    print(f"\nMigliore: lr = {best['lr']:.2e}, wd = {best['wd']:.2e}, mIoU = {best['miou']:.4f}")
    (OUT_DIR / "best_hparams.json").write_text(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
