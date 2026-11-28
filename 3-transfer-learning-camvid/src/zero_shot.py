"""Punto 3, variante 1: la FCN addestrata su Cityscapes valutata su CamVid SENZA addestramento.

Misura quanto la rete generalizza a un dominio diverso (Cambridge invece che città tedesche, altra camera,
altra risoluzione, etichette definite in modo un po' diverso). Nessun peso cambia: è solo inferenza.

La rete predice 19 classi Cityscapes; si convertono nelle 11 di CamVid (camvid_conf.CITYSCAPES_TO_CAMVID)
in modo "soft": somma delle probabilità delle classi accorpate, poi argmax sulle 11 (car + truck + bus +
train votano insieme per Car). La versione "hard" (argmax a 19 classi, poi tabella) dava lo stesso
risultato (+-0.0003 di mIoU, appunti_relazione.txt punto 17), quindi si tiene solo la soft.

Risultato principale alla risoluzione scelta per le varianti 2 e 3 (camvid_conf: 720x960, nativa):
metriche, confusion matrix e figura qualitativa. Le risoluzioni più basse di EVAL_SIZES servono solo al
grafico dell'IoU per classe (la rete ha imparato gli oggetti alla scala di Cityscapes 512x1024, e la scala
conta per una rete non riaddestrata).

Uso: python zero_shot.py   (risultati in outputs/es3_zero_shot/)
"""
import json

import matplotlib

matplotlib.use("Agg")  # salva i grafici su file senza aprire finestre
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

import camvid_conf as cconf
import camvid_metrics as metrics
from camvid_dataset import build_split
from camvid_engine import load_cityscapes_model
from camvid_utils import cityscapes_to_camvid_index, decode_segmap, to_camvid_soft

# ------------------------------------------------------------------ parametri
OUT_DIR = cconf.OUTPUT_DIR / "es3_zero_shot"
MAIN_SIZE = (cconf.IMG_HEIGHT, cconf.IMG_WIDTH)  # risoluzione dei training delle varianti 2 e 3
# (altezza, larghezza) delle immagini di test, dalla più piccola alla nativa (solo per il confronto)
EVAL_SIZES = [(360, 480), (540, 720), (720, 960)]
BATCH_SIZE = 2
QUALITATIVE_IDX = [0, 60, 120, 180]  # immagini di test mostrate nella figura (sparse lungo le sequenze)
# Rampa di un solo colore, dal chiaro allo scuro: la risoluzione è una grandezza ordinata
SIZE_COLORS = ["#b7d3f5", "#5b93dd", "#0d366b"]


@torch.no_grad()
def evaluate_size(model: torch.nn.Module, mapping: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Confusion matrix a 11 classi sul test di CamVid a height x width."""
    loader = build_split("test", height, width, batch_size=BATCH_SIZE)
    cm = torch.zeros(cconf.N_CLASSES, cconf.N_CLASSES, dtype=torch.long, device="cuda")
    for imgs, masks in tqdm(loader, desc=f"test {height}x{width}", leave=False):
        imgs, masks = imgs.cuda(non_blocking=True), masks.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(imgs)  # [B, 19, H, W]
        cm += metrics.confusion_matrix(to_camvid_soft(logits, mapping), masks, cconf.N_CLASSES, cconf.IGNORE_INDEX)
    return cm


def summarize(cm: torch.Tensor) -> dict:
    """Metriche di una confusion matrix (stesse funzioni del punto 1)."""
    cm_f = cm.double()
    return {
        "miou": metrics.mean_iou(cm),
        "mf1": metrics.mean_f1(cm),
        "pixel_acc": metrics.pixel_accuracy(cm),
        "iou_per_class": dict(zip(cconf.CLASS_NAMES, metrics.iou_per_class(cm).tolist())),
        "f1_per_class": dict(zip(cconf.CLASS_NAMES, metrics.f1_per_class(cm).tolist())),
        # pixel predetti / pixel veri: < 1 classe sotto-predetta, > 1 sovra-predetta
        "pred_gt_per_class": dict(zip(cconf.CLASS_NAMES, (cm_f.sum(0) / cm_f.sum(1)).tolist())),
        "confusion_matrix": cm.cpu().tolist(),
    }


def plot_iou_per_size(results: dict, path) -> None:
    """IoU per classe alle varie risoluzioni (barre raggruppate), mIoU nella legenda."""
    x = np.arange(cconf.N_CLASSES)
    width = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for i, (size, m) in enumerate(results.items()):
        ious = [m["iou_per_class"][c] for c in cconf.CLASS_NAMES]
        ax.bar(x + (i - (len(results) - 1) / 2) * width, ious, width * 0.92, color=SIZE_COLORS[i],
               label=f"{size} (mIoU {m['miou']:.3f})")
    ax.set_xticks(x, cconf.CLASS_NAMES)
    ax.set(ylabel="IoU", ylim=(0, 1.1),
           title="FCN-8s di Cityscapes su CamVid senza training: IoU per classe e risoluzione di test")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(ncol=len(results), loc="upper center", frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
def plot_qualitative(model: torch.nn.Module, mapping: torch.Tensor, height: int, width: int, path) -> None:
    """Una riga per immagine di test: immagine | ground truth | predizione convertita in classi CamVid."""
    ds = build_split("test", height, width, batch_size=1).dataset
    mean, std = np.array(cconf.IMAGENET_MEAN), np.array(cconf.IMAGENET_STD)
    fig, axes = plt.subplots(len(QUALITATIVE_IDX), 3, figsize=(12, 3 * len(QUALITATIVE_IDX)))
    for r, i in enumerate(QUALITATIVE_IDX):
        img, mask = ds[i]
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(img[None].cuda())
        pred = to_camvid_soft(logits, mapping)[0].cpu().numpy()
        panels = [np.clip(img.permute(1, 2, 0).numpy() * std + mean, 0, 1), decode_segmap(mask.numpy()),
                  decode_segmap(pred)]
        for c, (panel, title) in enumerate(zip(panels, ["immagine", "ground truth", "FCN Cityscapes"])):
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
    assert MAIN_SIZE in EVAL_SIZES, f"La risoluzione principale {MAIN_SIZE} deve essere in EVAL_SIZES"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Rete: {cconf.CITYSCAPES_MODEL} da {cconf.CITYSCAPES_CKPT} | dati: {cconf.DATA_ROOT} | output: {OUT_DIR}")
    model = load_cityscapes_model().cuda().eval()  # 19 classi in uscita
    mapping = cityscapes_to_camvid_index(cconf.CITYSCAPES_CLASS_NAMES)

    results = {}
    for height, width in EVAL_SIZES:
        m = summarize(evaluate_size(model, mapping, height, width))
        results[f"{height}x{width}"] = m
        print(f"{height}x{width} | mIoU {m['miou']:.4f} | mF1 {m['mf1']:.4f} | pixel acc {m['pixel_acc']:.4f}")
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    plot_iou_per_size(results, OUT_DIR / "iou_per_class_per_size.png")

    # Risultato principale: alla risoluzione dei training
    height, width = MAIN_SIZE
    key = f"{height}x{width}"
    main_res = results[key]
    print(f"\nRisoluzione principale {key}: mIoU {main_res['miou']:.4f} | mF1 {main_res['mf1']:.4f} "
          f"| pixel acc {main_res['pixel_acc']:.4f}")
    for name in cconf.CLASS_NAMES:
        print(f"    {name:12s} IoU {main_res['iou_per_class'][name]:.3f} | F1 {main_res['f1_per_class'][name]:.3f} "
              f"| pred/gt {main_res['pred_gt_per_class'][name]:.2f}")
    (OUT_DIR / "test_metrics.json").write_text(json.dumps(main_res, indent=2))
    plot_confusion(main_res["confusion_matrix"],
                   f"FCN Cityscapes su CamVid, senza training ({key}, mIoU {main_res['miou']:.3f})",
                   OUT_DIR / f"confusion_{key}.png")
    plot_qualitative(model, mapping, height, width, OUT_DIR / f"qualitative_{key}.png")
    print(f"Risultati e figure in {OUT_DIR}")


if __name__ == "__main__":
    main()
