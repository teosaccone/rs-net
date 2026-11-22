"""Figure e tempi di inferenza per la relazione, a partire dalle run già finite.

Uso: python figures.py   (risultati in outputs/figures/)

Produce:
  val_miou.png            mIoU di validazione per epoca, tutte le run sovrapposte
  iou_per_class.png       IoU di test per classe, run affiancate
  recall_pred_gt.png      per classe: recall e pixel predetti / pixel veri (sotto 1 = classe sotto-predetta)
  confusion_<run>.png     confusion matrix di test normalizzata per riga (diagonale = recall)
  qualitative.png         immagine | ground truth | predizione di ogni run, su immagini di test
  qualitative_zoom.png    come sopra, ritagliato sulla zona con più pali/semafori/cartelli
  inference.json          parametri e tempo di inferenza per immagine (512x1024, batch 1, AMP)
"""
import csv
import json
import time

import matplotlib

matplotlib.use("Agg")  # salva i grafici su file senza aprire finestre
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

import conf
from dataset import CityscapeDataset, split_train_val_test
from engine import MODELS
from transforms import get_val_transforms
from utils import decode_segmap, encode_mask

# ------------------------------------------------------------------ parametri
OUT_DIR = conf.OUTPUT_DIR / "figures"

# Nome in figura -> (nome del modello in engine.MODELS, cartella della run)
RUNS = {
    "FCN-32s": ("fcn32s", conf.OUTPUT_DIR / "train"),  # run A (cartella con il nome vecchio)
    "FCN-16s": ("fcn16s", conf.OUTPUT_DIR / "fcn16s_ce"),
    "FCN-16s + Dice": ("fcn16s", conf.OUTPUT_DIR / "fcn16s_ce_dice"),  # run D: loss CE + Dice
}
# Un colore fisso per run (palette categoriale: blu, arancio, acqua, giallo)
RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

N_QUALITATIVE = 4              # immagini di test nelle figure qualitative
THIN_CLASSES = [5, 6, 7]       # pole, traffic light, traffic sign: dove si vede l'effetto dello skip
ZOOM_SIZE = (192, 384)         # ritaglio delle figure zoom (sull'immagine 512x1024)

TIMING_WARMUP, TIMING_STEPS = 10, 50


# ------------------------------------------------------------------ grafici dai file della run
def plot_val_miou() -> None:
    """mIoU di validazione per epoca, una linea per run (da log.csv)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for (name, (_, run_dir)), color in zip(RUNS.items(), RUN_COLORS):
        with open(run_dir / "log.csv", newline="") as f:
            log = list(csv.DictReader(f))
        epochs = [int(row["epoch"]) for row in log]
        mious = [float(row["val_miou"]) for row in log]
        ax.plot(epochs, mious, color=color, linewidth=2, label=name)
        best = int(np.argmax(mious))
        ax.plot(epochs[best], mious[best], "o", color=color, markersize=8)
    ax.set(xlabel="epoca", ylabel="mIoU di validazione", title="mIoU di validazione (pallino = best.pth)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "val_miou.png", dpi=150)
    plt.close(fig)


def load_test_metrics() -> dict[str, dict]:
    return {name: json.loads((run_dir / "test_metrics.json").read_text()) for name, (_, run_dir) in RUNS.items()}


def plot_iou_per_class(metrics: dict[str, dict]) -> None:
    """Barre affiancate dell'IoU di test per classe, una barra per run."""
    n_runs = len(metrics)
    width = 0.8 / n_runs
    x = np.arange(conf.N_CLASSES)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for i, ((name, m), color) in enumerate(zip(metrics.items(), RUN_COLORS)):
        ious = [m["iou_per_class"][c] for c in conf.CLASS_NAMES]
        ax.bar(x + (i - (n_runs - 1) / 2) * width, ious, width * 0.9, color=color,
               label=f"{name} (mIoU {m['miou']:.3f})")
    ax.set_xticks(x, conf.CLASS_NAMES, rotation=45, ha="right")
    ax.set(ylabel="IoU di test", ylim=(0, 1), title="IoU per classe sul test set")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "iou_per_class.png", dpi=150)
    plt.close(fig)


def plot_recall_pred_gt(metrics: dict[str, dict]) -> None:
    """Per classe, una barra per run: recall (sopra) e pixel predetti / pixel veri (sotto).

    pred/gt < 1: la rete disegna la classe meno di quanto c'è davvero (sotto-predetta), > 1: troppo.
    Serve a vedere l'effetto del Dice sulle classi piccole, che la sola CE sotto-predice.
    """
    n_runs = len(metrics)
    width = 0.8 / n_runs
    x = np.arange(conf.N_CLASSES)
    fig, (ax_rec, ax_ratio) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    for i, ((name, m), color) in enumerate(zip(metrics.items(), RUN_COLORS)):
        cm = np.array(m["confusion_matrix"], dtype=float)  # cm[vera][predetta]
        tp, gt, pred = np.diag(cm), cm.sum(axis=1), cm.sum(axis=0)
        offset = (i - (n_runs - 1) / 2) * width
        ax_rec.bar(x + offset, tp / gt, width * 0.9, color=color, label=name)
        ax_ratio.bar(x + offset, pred / gt, width * 0.9, color=color, label=name)
    ax_rec.set(ylabel="recall di test", ylim=(0, 1.15), title="Recall e pixel predetti / pixel veri per classe (test)")
    ax_ratio.axhline(1.0, color="black", linewidth=1)  # 1 = la classe è predetta tanto quanto c'è
    ax_ratio.set(ylabel="pixel predetti / pixel veri")
    ax_ratio.set_xticks(x, conf.CLASS_NAMES, rotation=45, ha="right")
    for ax in (ax_rec, ax_ratio):
        ax.grid(axis="y", alpha=0.3)
    ax_rec.legend(ncol=n_runs, loc="upper center")  # in una riga sopra le barre (che arrivano a 1)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "recall_pred_gt.png", dpi=150)
    plt.close(fig)


def plot_confusion(name: str, m: dict) -> None:
    """Confusion matrix normalizzata per riga: riga = classe vera, colonna = classe predetta."""
    cm = np.array(m["confusion_matrix"], dtype=float)
    cm = cm / cm.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("blues", ["#ffffff", "#86b6ef", "#256abf", "#0d366b"])
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
    for t in range(conf.N_CLASSES):
        for p in range(conf.N_CLASSES):
            if cm[t, p] >= 0.05:  # solo i valori rilevanti, per leggibilità
                ax.text(p, t, f"{cm[t, p]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cm[t, p] > 0.5 else "black")
    ax.set_xticks(range(conf.N_CLASSES), conf.CLASS_NAMES, rotation=90)
    ax.set_yticks(range(conf.N_CLASSES), conf.CLASS_NAMES)
    ax.set(xlabel="classe predetta", ylabel="classe vera",
           title=f"{name}: confusion matrix di test normalizzata per riga (mIoU {m['miou']:.3f})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    # Nome del file dal nome della run: "FCN-16s + Dice" -> confusion_fcn16s_dice.png
    fig.savefig(OUT_DIR / f"confusion_{name.lower().replace('-', '').replace(' + ', '_')}.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ modelli
def load_model(model_name: str, run_dir) -> torch.nn.Module:
    """Rete con i pesi di best.pth, in eval su GPU."""
    model = MODELS[model_name](pretrained=False)
    model.load_state_dict(torch.load(run_dir / "best.pth", map_location="cpu", weights_only=True))
    return model.cuda().eval()


@torch.no_grad()
def measure_inference(model: torch.nn.Module) -> float:
    """Millisecondi per immagine 512x1024 (batch 1, AMP), dopo qualche passo di riscaldamento."""
    x = torch.randn(1, 3, conf.IMG_HEIGHT, conf.IMG_WIDTH, device="cuda")
    for step in range(TIMING_WARMUP + TIMING_STEPS):
        if step == TIMING_WARMUP:
            torch.cuda.synchronize()  # la GPU lavora in modo asincrono: si aspetta che finisca
            start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            model(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / TIMING_STEPS * 1000


@torch.no_grad()
def predict(model: torch.nn.Module, img: torch.Tensor) -> np.ndarray:
    with torch.autocast("cuda", dtype=torch.float16):
        return model(img[None].cuda()).argmax(dim=1)[0].cpu().numpy()


# ------------------------------------------------------------------ figure qualitative
def pick_test_images(masks: list) -> list[int]:
    """Indici delle immagini di test con più pixel di pali, semafori e cartelli."""
    counts = [np.isin(encode_mask(np.array(Image.open(m))), THIN_CLASSES).sum() for m in masks]
    return [int(i) for i in np.argsort(counts)[::-1][:N_QUALITATIVE]]


def zoom_box(mask: np.ndarray) -> tuple[slice, slice]:
    """Ritaglio ZOOM_SIZE centrato sui pixel delle classi sottili (dentro i bordi dell'immagine)."""
    ys, xs = np.nonzero(np.isin(mask, THIN_CLASSES))
    h, w = ZOOM_SIZE
    top = int(np.clip(np.median(ys) - h // 2, 0, mask.shape[0] - h))
    left = int(np.clip(np.median(xs) - w // 2, 0, mask.shape[1] - w))
    return slice(top, top + h), slice(left, left + w)


def plot_qualitative(rows: list[dict], zoom: bool) -> None:
    """Una riga per immagine: immagine | ground truth | una colonna per run."""
    cols = ["immagine", "ground truth"] + list(RUNS)
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.6 * len(cols), 2.1 * len(rows)))
    for r, row in enumerate(rows):
        box = zoom_box(row["mask"]) if zoom else (slice(None), slice(None))
        panels = [row["image"], decode_segmap(row["mask"])] + [decode_segmap(p) for p in row["preds"]]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel[box])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c])
    fig.suptitle("Predizioni sul test set" + (" (ingrandimento su pali, semafori, cartelli)" if zoom else "")
                 + "; nero = pixel ignorati")
    fig.tight_layout()
    fig.savefig(OUT_DIR / ("qualitative_zoom.png" if zoom else "qualitative.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Grafici dai file già salvati (log.csv, test_metrics.json): niente GPU
    plot_val_miou()
    metrics = load_test_metrics()
    plot_iou_per_class(metrics)
    plot_recall_pred_gt(metrics)
    for name, m in metrics.items():
        plot_confusion(name, m)

    # 2. Immagini di test per le figure qualitative (stesse per tutte le run)
    _, _, test = split_train_val_test()
    idx = pick_test_images(test[1])
    print("Immagini di test scelte:", [test[0][i].name for i in idx])
    ds = CityscapeDataset(*test, get_val_transforms(conf.IMAGENET_MEAN, conf.IMAGENET_STD))
    mean, std = np.array(conf.IMAGENET_MEAN), np.array(conf.IMAGENET_STD)
    rows = []
    for i in idx:
        img, mask = ds[i]
        image = np.clip(img.permute(1, 2, 0).numpy() * std + mean, 0, 1)  # denormalizzata per mostrarla
        rows.append({"img": img, "image": image, "mask": mask.numpy(), "preds": []})

    # 3. Una run alla volta (in 4 GB non stanno due reti insieme): predizioni e tempo di inferenza
    inference = {}
    for name, (model_name, run_dir) in RUNS.items():
        model = load_model(model_name, run_dir)
        for row in rows:
            row["preds"].append(predict(model, row["img"]))
        ms = measure_inference(model)
        inference[name] = {
            "params_total": sum(p.numel() for p in model.parameters()),
            "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "ms_per_image": ms,
            "images_per_s": 1000 / ms,
        }
        print(f"{name}: {ms:.1f} ms per immagine ({1000 / ms:.1f} img/s)")
        del model
        torch.cuda.empty_cache()
    (OUT_DIR / "inference.json").write_text(json.dumps(inference, indent=2))

    plot_qualitative(rows, zoom=False)
    plot_qualitative(rows, zoom=True)
    print(f"Figure salvate in {OUT_DIR}")


if __name__ == "__main__":
    main()
