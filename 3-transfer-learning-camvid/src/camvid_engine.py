"""Copia di engine.py del punto 1 adattata a CamVid (11 classi, IGNORE_INDEX di camvid_conf).

Uguali al punto 1: dice_loss, compute_loss, train_one_epoch (AMP, clipping, accumulo, poly per passo),
evaluate. Nuovo: build_model sceglie da dove partono i pesi (camvid_conf.INIT):
  "imagenet"   variante 2: VGG16 ImageNet + classificatori a zero con 11 uscite
  "cityscapes" variante 3: tutti i pesi della FCN di Cityscapes, classificatori ridotti da 19 a 11 classi
"""
import math

import torch
import torch.nn.functional as F
from tqdm import tqdm

import camvid_conf as conf
from camvid_fcn import MODELS
from camvid_metrics import confusion_matrix, iou_per_class, mean_f1, mean_iou, pixel_accuracy
from camvid_utils import seed_everything

N_CITYSCAPES_CLASSES = 19


# ------------------------------------------------------------------ modello
def load_cityscapes_model(model_name: str = conf.CITYSCAPES_MODEL) -> torch.nn.Module:
    """FCN di Cityscapes (19 classi) con i pesi di best.pth del punto 1, su CPU."""
    model = MODELS[model_name](N_CITYSCAPES_CLASSES, pretrained=False)  # init casuale, poi sovrascritta tutta
    model.load_state_dict(torch.load(conf.CITYSCAPES_CKPT, map_location="cpu", weights_only=True))
    return model


def representative_indices() -> torch.Tensor:
    """Tensore [11]: per ogni classe CamVid l'indice (trainId) della classe Cityscapes rappresentante."""
    return torch.tensor([conf.CITYSCAPES_CLASS_NAMES.index(conf.CITYSCAPES_REPRESENTATIVE[name])
                         for name in conf.CLASS_NAMES])


def transfer_cityscapes_weights(model: torch.nn.Module, cityscapes_state: dict[str, torch.Tensor]) -> None:
    """Copia nella rete a 11 classi tutti i pesi della rete Cityscapes (variante 3).

    - Backbone e fc6/fc7: identici (stesse shape), copiati così come sono.
    - Classificatori (score_fr, score_pool4, score_pool3: [19, C, 1, 1] e bias [19]): per ogni classe CamVid
      si copia la riga della classe Cityscapes rappresentante (Car <- car, Building <- building, ...).
      Sommare le righe di un gruppo (car + truck + bus + train) sarebbe sbagliato: si sommerebbero i LOGIT,
      non le probabilità, e un logit molto negativo di "train" abbasserebbe quello di Car.
    - Upsampling ([19, 19, k, k], canale di ingresso x canale di uscita): si tiene la sottomatrice delle
      classi rappresentanti. Quelli congelati sono bilineari (diagonali), la sottomatrice è ancora bilineare;
      quelli allenati (upscore2, upscore_pool4) dopo Cityscapes sono quasi diagonali (energia fuori
      diagonale 0.6% e 1.7%), quindi si perde pochissimo.
    All'inizio la rete predice quindi come quella di Cityscapes, limitata alle 11 classi rappresentanti.
    """
    rep = representative_indices()
    new_state = {}
    for key, value in model.state_dict().items():
        src = cityscapes_state[key]
        if src.shape == value.shape:
            new_state[key] = src
        elif "upscore" in key:          # [19, 19, k, k] -> [11, 11, k, k]
            new_state[key] = src[rep][:, rep]
        else:                           # classificatori: [19, ...] -> [11, ...]
            new_state[key] = src[rep]
        assert new_state[key].shape == value.shape, f"{key}: {tuple(new_state[key].shape)} != {tuple(value.shape)}"
    model.load_state_dict(new_state)


def create_model(init: str = conf.INIT, model_name: str = conf.MODEL) -> torch.nn.Module:
    """Rete a 11 classi su CPU, con i pesi di partenza della variante scelta ("imagenet" o "cityscapes")."""
    if init == "imagenet":
        return MODELS[model_name](conf.N_CLASSES, pretrained=True)
    if init == "cityscapes":
        assert model_name == conf.CITYSCAPES_MODEL, "La rete di Cityscapes salvata è una " + conf.CITYSCAPES_MODEL
        model = MODELS[model_name](conf.N_CLASSES, pretrained=False)  # init casuale, poi sovrascritta tutta
        transfer_cityscapes_weights(model, torch.load(conf.CITYSCAPES_CKPT, map_location="cpu", weights_only=True))
        return model
    raise ValueError(f"INIT sconosciuto: {init}")


def build_model(lr: float, wd: float, dropout: bool = True, momentum: float = 0.9,
                init: str = conf.INIT, model_name: str = conf.MODEL):
    """Rete + SGD + GradScaler, sempre con lo stesso seed (come build_model del punto 1)."""
    seed_everything(conf.SEED)
    model = create_model(init, model_name).cuda()
    if not dropout:
        model.head.drop6.p = model.head.drop7.p = 0.0
    params = [p for p in model.parameters() if p.requires_grad]  # gli upsampling congelati restano fuori
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
    return model, optimizer, torch.amp.GradScaler("cuda")


# ------------------------------------------------------------------ loss (identiche al punto 1)
def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice loss = 1 - Dice medio sulle classi presenti nel batch (pixel IGNORE_INDEX esclusi).

    Dice_c = 2 * somma(p_c * g_c) / (somma(p_c) + somma(g_c)), somme su tutto il batch, softmax in float32.
    """
    probs = logits.float().softmax(dim=1)
    valid = targets != conf.IGNORE_INDEX
    one_hot = F.one_hot(targets.masked_fill(~valid, 0), probs.size(1)).permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1)
    probs, one_hot = probs * valid, one_hot * valid

    dims = (0, 2, 3)  # una somma per classe
    intersection = (probs * one_hot).sum(dims)
    dice = (2 * intersection + eps) / (probs.sum(dims) + one_hot.sum(dims) + eps)
    present = one_hot.sum(dims) > 0
    if not present.any():  # batch con soli pixel ignorati
        return probs.sum() * 0.0
    return 1 - dice[present].mean()


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, dice_weight: float) -> torch.Tensor:
    """Cross-entropy (pixel ignore esclusi) + dice_weight * Dice. Con dice_weight = 0 è la sola CE."""
    loss = F.cross_entropy(logits.float(), targets, ignore_index=conf.IGNORE_INDEX)
    if dice_weight > 0:
        loss = loss + dice_weight * dice_loss(logits, targets)
    return loss


# ------------------------------------------------------------------ training e valutazione (identici al punto 1)
def train_one_epoch(model, loader, optimizer, scaler, dice_weight, max_steps=None, ref_loss=None,
                    scheduler=None, max_grad_norm=None, grad_norms=None, accum_steps=1,
                    cm=None, desc=None):
    """Restituisce (loss di ogni micro-batch, esplosa?). Parametri come in engine.py del punto 1:
    scheduler aggiornato a ogni passo, clipping a max_grad_norm, norme (prima del taglio) in grad_norms,
    gradient accumulation su accum_steps micro-batch, confusion matrix di train in cm, tqdm se desc.
    Si ferma se la loss è NaN o supera 3 volte ref_loss.
    """
    model.train()
    losses = []
    optimizer.zero_grad()
    total = len(loader) if max_steps is None else min(max_steps, len(loader))
    pbar = tqdm(loader, total=total, desc=desc, disable=desc is None, dynamic_ncols=True, leave=False)
    for step, (imgs, masks) in enumerate(pbar):
        if step == max_steps:
            break
        imgs, masks = imgs.cuda(), masks.cuda().long()

        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(imgs)
            loss = compute_loss(logits, masks, dice_weight)

        losses.append(loss.item())
        ref_loss = ref_loss or losses[0]
        if not math.isfinite(losses[-1]) or losses[-1] > 3 * ref_loss:
            pbar.close()
            return losses, True
        if cm is not None:
            cm += confusion_matrix(logits.argmax(dim=1), masks, conf.N_CLASSES, conf.IGNORE_INDEX)
        if step % 10 == 0:
            pbar.set_postfix(loss=f"{sum(losses[-50:]) / len(losses[-50:]):.3f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        # loss / accum_steps: la somma dei gradienti diventa la media sul batch effettivo
        scaler.scale(loss / accum_steps).backward()
        if (step + 1) % accum_steps != 0:
            continue

        if max_grad_norm is not None or grad_norms is not None:
            scaler.unscale_(optimizer)  # gradienti al valore vero prima di misurarli / tagliarli
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm or float("inf"))
            if grad_norms is not None:
                grad_norms.append(norm.item())
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
    pbar.close()
    return losses, False


@torch.no_grad()
def evaluate(model, loader, dice_weight, desc=None) -> dict:
    """Loss media (pesata per pixel validi), mIoU, F1 medio, pixel accuracy, IoU per classe e confusion matrix."""
    model.eval()
    cm = torch.zeros((conf.N_CLASSES, conf.N_CLASSES), dtype=torch.long, device="cuda")
    loss_sum, n_pixels = 0.0, 0
    for imgs, masks in tqdm(loader, desc=desc, disable=desc is None, dynamic_ncols=True, leave=False):
        imgs, masks = imgs.cuda(), masks.cuda().long()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(imgs)
        n_valid = int((masks != conf.IGNORE_INDEX).sum())
        if n_valid > 0:
            loss_sum += compute_loss(logits, masks, dice_weight).item() * n_valid
            n_pixels += n_valid
        cm += confusion_matrix(logits.argmax(dim=1), masks, conf.N_CLASSES, conf.IGNORE_INDEX)
    return {
        "loss": loss_sum / max(n_pixels, 1),
        "miou": mean_iou(cm),
        "f1": mean_f1(cm),
        "pixel_acc": pixel_accuracy(cm),
        "iou_per_class": iou_per_class(cm).tolist(),
        "confusion_matrix": cm.tolist(),
    }
