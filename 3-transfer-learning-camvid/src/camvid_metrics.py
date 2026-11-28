"""Copia di metrics.py del punto 1 (identica). Metriche di segmentazione calcolate a partire dalla matrice di confusione.

Convenzione: cm[t, p] = numero di pixel con classe vera t predetti come p.
I pixel con ignore_index sono esclusi prima di costruire la matrice.
"""
import torch


def confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Matrice di confusione [n_classes, n_classes] (int64) da predizioni e target della stessa shape."""
    valid = targets != ignore_index
    t = targets[valid].long()
    p = preds[valid].long()
    # Ogni coppia (t, p) diventa un indice unico t*C + p: un solo bincount conta tutte le coppie
    counts = torch.bincount(t * n_classes + p, minlength=n_classes * n_classes)
    return counts.reshape(n_classes, n_classes)


def _tp_fp_fn(cm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cm = cm.double()
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp  # predetti come c ma di un'altra classe
    fn = cm.sum(dim=1) - tp  # di classe c ma predetti come altro
    return tp, fp, fn


def iou_per_class(cm: torch.Tensor) -> torch.Tensor:
    """IoU = TP / (TP + FP + FN). NaN per le classi assenti sia nel target sia nelle predizioni."""
    tp, fp, fn = _tp_fp_fn(cm)
    return tp / (tp + fp + fn)  # 0/0 -> NaN: classe da non contare nella media


def f1_per_class(cm: torch.Tensor) -> torch.Tensor:
    """F1 (= Dice) = 2TP / (2TP + FP + FN). NaN per le classi assenti."""
    tp, fp, fn = _tp_fp_fn(cm)
    return 2 * tp / (2 * tp + fp + fn)


def mean_iou(cm: torch.Tensor) -> float:
    """Media dell'IoU sulle sole classi presenti (ignora i NaN)."""
    return torch.nanmean(iou_per_class(cm)).item()


def mean_f1(cm: torch.Tensor) -> float:
    return torch.nanmean(f1_per_class(cm)).item()


def pixel_accuracy(cm: torch.Tensor) -> float:
    """Frazione di pixel validi classificati correttamente."""
    total = cm.sum()
    return (cm.diag().sum() / total).item() if total > 0 else float("nan")
