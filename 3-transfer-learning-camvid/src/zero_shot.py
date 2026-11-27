"""Punto 3, variante 1: la FCN addestrata su Cityscapes valutata su CamVid SENZA addestramento.

Misura quanto la rete generalizza a un dominio diverso (Cambridge invece che città tedesche, altra camera,
altra risoluzione, etichette definite in modo un po' diverso). Nessun peso cambia: è solo inferenza.

La rete predice 19 classi Cityscapes; si convertono nelle 11 di CamVid (camvid_conf.CITYSCAPES_TO_CAMVID)
in due modi, entrambi riportati:
  hard: argmax sulle 19 classi, poi conversione (es. vince "truck" -> Car)
  soft: somma delle probabilità delle classi accorpate, poi argmax sulle 11 (car + truck + bus + train
        votano insieme per Car)
La valutazione si ripete a più risoluzioni (EVAL_SIZES): la rete ha imparato oggetti alla scala delle
immagini Cityscapes 512x1024, e la scala conta molto per una rete non riaddestrata.

Uso: python zero_shot.py   (risultati in outputs/es3_zero_shot/)
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")  # salva i grafici su file senza aprire finestre
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

import camvid_conf as cconf

sys.path.insert(0, str(cconf.FCN_SRC))  # reti FCN, metriche e conf di Cityscapes del punto 1
import conf as cityscapes_conf  # noqa: E402  (classi Cityscapes, nell'ordine dei trainId)
import metrics  # noqa: E402
from engine import MODELS  # noqa: E402

from camvid_dataset import build_split  # noqa: E402
from camvid_utils import cityscapes_to_camvid_index, decode_segmap, to_camvid_hard, to_camvid_soft  # noqa: E402

# ------------------------------------------------------------------ parametri
OUT_DIR = cconf.OUTPUT_DIR / "es3_zero_shot"
# (altezza, larghezza) in cui si ridimensionano le immagini di test (CamVid nativo: 720 x 960)
EVAL_SIZES = [(360, 480), (540, 720), (720, 960)]
BATCH_SIZE = 2
METHODS = {"hard": to_camvid_hard, "soft": to_camvid_soft}
QUALITATIVE_IDX = [0, 60, 120, 180]  # immagini di test mostrate nella figura (sparse lungo le sequenze)


def load_cityscapes_model() -> torch.nn.Module:
    """FCN di Cityscapes con i pesi di best.pth, in eval su GPU (19 classi in uscita)."""
    model = MODELS[cconf.CITYSCAPES_MODEL](pretrained=False)
    model.load_state_dict(torch.load(cconf.CITYSCAPES_CKPT, map_location="cpu", weights_only=True))
    return model.cuda().eval()


@torch.no_grad()
def evaluate_size(model: torch.nn.Module, mapping: torch.Tensor, height: int, width: int) -> dict[str, torch.Tensor]:
    """Confusion matrix a 11 classi sul test di CamVid, una per metodo di conversione."""
    loader = build_split("test", height, width, batch_size=BATCH_SIZE)
    cms = {name: torch.zeros(cconf.N_CLASSES, cconf.N_CLASSES, dtype=torch.long, device="cuda") for name in METHODS}
    for imgs, masks in tqdm(loader, desc=f"test {height}x{width}", leave=False):
        imgs, masks = imgs.cuda(non_blocking=True), masks.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(imgs)  # [B, 19, H, W]
        for name, convert in METHODS.items():
            cms[name] += metrics.confusion_matrix(convert(logits, mapping), masks, cconf.N_CLASSES, cconf.IGNORE_INDEX)
    return cms


def summarize(cm: torch.Tensor) -> dict:
    """Metriche di una confusion matrix (stesse funzioni del punto 1)."""
    return {
        "miou": metrics.mean_iou(cm),
        "mf1": metrics.mean_f1(cm),
        "pixel_acc": metrics.pixel_accuracy(cm),
        "iou_per_class": dict(zip(cconf.CLASS_NAMES, metrics.iou_per_class(cm).tolist())),
        "f1_per_class": dict(zip(cconf.CLASS_NAMES, metrics.f1_per_class(cm).tolist())),
        "confusion_matrix": cm.cpu().tolist(),
    }


def plot_confusion(cm: list, title: str, path) -> None:
    """Confusion matrix normalizzata per riga (riga = classe vera, diagonale = recall)."""
    cm = np.array(cm, dtype=float)
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list("blues", ["#ffffff", "#86b6ef", "#256abf", "#0d366b"])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
    for t in range(cconf.N_CLASSES):
        for p in range(cconf.N_CLASSES):
            if cm[t, p] >= 0.05:  # solo i valori rilevanti, per leggibilità
                ax.text(p, t, f"{cm[t, p]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[t, p] > 0.5 else "black")
    ax.set_xticks(range(cconf.N_CLASSES), cconf.CLASS_NAMES, rotation=90)
    ax.set_yticks(range(cconf.N_CLASSES), cconf.CLASS_NAMES)
    ax.set(xlabel="classe predetta", ylabel="classe vera", title=title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_qualitative(model: torch.nn.Module, mapping: torch.Tensor, height: int, width: int, method: str, path) -> None:
    """Una riga per immagine di test: immagine | ground truth | predizione convertita in classi CamVid."""
    ds = build_split("test", height, width, batch_size=1).dataset
    mean, std = np.array(cconf.IMAGENET_MEAN), np.array(cconf.IMAGENET_STD)
    fig, axes = plt.subplots(len(QUALITATIVE_IDX), 3, figsize=(12, 3 * len(QUALITATIVE_IDX)))
    for r, i in enumerate(QUALITATIVE_IDX):
        img, mask = ds[i]
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(img[None].cuda())
        pred = METHODS[method](logits, mapping)[0].cpu().numpy()
        panels = [np.clip(img.permute(1, 2, 0).numpy() * std + mean, 0, 1), decode_segmap(mask.numpy()),
                  decode_segmap(pred)]
        for c, (panel, title) in enumerate(zip(panels, ["immagine", "ground truth", f"FCN Cityscapes ({method})"])):
            axes[r, c].imshow(panel)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(title)
    fig.suptitle(f"CamVid test senza addestramento, {height}x{width}; nero = pixel ignorati (Void)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Rete: {cconf.CITYSCAPES_MODEL} da {cconf.CITYSCAPES_CKPT} | dati: {cconf.DATA_ROOT} | output: {OUT_DIR}")
    model = load_cityscapes_model()
    mapping = cityscapes_to_camvid_index(cityscapes_conf.CLASS_NAMES)

    results = {}
    for height, width in EVAL_SIZES:
        cms = evaluate_size(model, mapping, height, width)
        for name, cm in cms.items():
            results[f"{height}x{width}_{name}"] = summarize(cm)
            m = results[f"{height}x{width}_{name}"]
            print(f"{height}x{width} {name:4s} | mIoU {m['miou']:.4f} | mF1 {m['mf1']:.4f} | pixel acc {m['pixel_acc']:.4f}")
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))

    # Configurazione migliore: IoU per classe, confusion matrix e immagini
    best = max(results, key=lambda k: results[k]["miou"])
    print(f"\nMigliore: {best} (mIoU {results[best]['miou']:.4f})")
    for name, iou in results[best]["iou_per_class"].items():
        print(f"    {name:12s} IoU {iou:.3f}")
    size, method = best.split("_")
    height, width = map(int, size.split("x"))
    plot_confusion(results[best]["confusion_matrix"],
                   f"FCN Cityscapes su CamVid, senza training ({best}, mIoU {results[best]['miou']:.3f})",
                   OUT_DIR / f"confusion_{best}.png")
    plot_qualitative(model, mapping, height, width, method, OUT_DIR / f"qualitative_{best}.png")
    print(f"Risultati e figure in {OUT_DIR}")


if __name__ == "__main__":
    main()
