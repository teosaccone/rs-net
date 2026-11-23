"""Costruzione del modello, loss, un'epoca di training e la valutazione, con mixed precision (AMP) su GPU."""
import math

import torch
import torch.nn.functional as F
from tqdm import tqdm

import conf
from FCN16s import FCN16s
from FCN32s import FCN32s
from FCN8s import FCN8s
from metrics import confusion_matrix, iou_per_class, mean_f1, mean_iou, pixel_accuracy
from utils import seed_everything

# Nome (come in conf.MODEL) -> classe della rete
MODELS = {"fcn32s": FCN32s, "fcn16s": FCN16s, "fcn8s": FCN8s}


def build_model(lr: float, wd: float, dropout: bool = True, momentum: float = 0.9,
                model_name: str = conf.MODEL, pretrained: bool = conf.PRETRAINED):
    """Rete (pre-addestrata su ImageNet o da zero) + SGD + GradScaler, sempre con lo stesso seed
    (stessa inizializzazione, anche dei pesi casuali)."""
    seed_everything(conf.SEED)
    model = MODELS[model_name](pretrained=pretrained).cuda()
    if not dropout:
        model.head.drop6.p = model.head.drop7.p = 0.0
    params = [p for p in model.parameters() if p.requires_grad]  # gli upsampling congelati restano fuori
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
    return model, optimizer, torch.amp.GradScaler("cuda")


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice loss = 1 - Dice medio sulle classi presenti nel batch.

    "Soft": usa le probabilità della softmax invece dell'argmax, così è derivabile.
    Per ogni classe c, sommando su tutti i pixel validi del batch:
        Dice_c = 2 * somma(p_c * g_c) / (somma(p_c) + somma(g_c))
    con p_c = probabilità predetta di c e g_c = 1 se il pixel è di classe c (one-hot).
    - I pixel IGNORE_INDEX sono esclusi (azzerati sia in p sia in g).
    - Somme su tutto il batch, non immagine per immagine: più stabile per le classi con pochi pixel.
    - Media solo sulle classi presenti nel target del batch: per le assenti sarebbe 0/0.
      Gli errori su di loro (predirle dove non ci sono) li penalizza la cross-entropy.
    - eps evita divisioni per zero; con milioni di pixel è trascurabile.
    """
    probs = logits.float().softmax(dim=1)  # float32 anche con AMP: le somme su milioni di pixel in fp16 perdono precisione
    valid = targets != conf.IGNORE_INDEX
    one_hot = F.one_hot(targets.masked_fill(~valid, 0), probs.size(1)).permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1)  # [B, 1, H, W]: si moltiplica su tutte le classi
    probs, one_hot = probs * valid, one_hot * valid

    dims = (0, 2, 3)  # somma su batch e pixel, una somma per classe
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


def train_one_epoch(model, loader, optimizer, scaler, dice_weight, max_steps=None, ref_loss=None,
                    scheduler=None, max_grad_norm=None, grad_norms=None, accum_steps=1,
                    cm=None, desc=None):
    """Restituisce (loss di ogni micro-batch, esplosa?).

    dice_weight: peso del Dice nella loss (0 = sola cross-entropy), vedi compute_loss.

    Si ferma se la loss diventa NaN o supera 3 volte la loss iniziale ref_loss (slide 9).
    scheduler: se passato, la lr viene aggiornata a ogni passo dell'optimizer.
    max_grad_norm: se passato, la norma del gradiente viene tagliata a questo valore.
    grad_norms: se passata una lista, ci aggiunge la norma del gradiente (prima del taglio) di ogni passo.
    accum_steps: gradient accumulation. I gradienti di accum_steps micro-batch si sommano e i pesi
        si aggiornano una volta sola: equivale a un batch accum_steps volte più grande
        (esattamente, perché la rete non ha BatchNorm). I micro-batch avanzati a fine epoca si scartano.
    cm: se passata una confusion matrix (C x C su cuda), ci accumula le predizioni di train
        (per il mIoU di train: costa solo un argmax, le predizioni ci sono già).
    desc: se passato, mostra una progress bar tqdm con questa descrizione.
    """
    model.train()
    losses = []
    optimizer.zero_grad()
    total = len(loader) if max_steps is None else min(max_steps, len(loader))
    pbar = tqdm(loader, total=total, desc=desc, disable=desc is None, dynamic_ncols=True, leave=False)
    for step, (imgs, masks) in enumerate(pbar):
        if step == max_steps:
            break
        imgs, masks = imgs.cuda(), masks.cuda().long()  # CrossEntropy vuole target int64

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
        if step % 10 == 0:  # media mobile delle ultime 50 loss: più leggibile del valore singolo
            pbar.set_postfix(loss=f"{sum(losses[-50:]) / len(losses[-50:]):.3f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        # loss divisa per accum_steps: la somma dei gradienti diventa la loro media sul batch effettivo.
        # Scalata dall'AMP per evitare underflow dei gradienti in fp16.
        scaler.scale(loss / accum_steps).backward()
        if (step + 1) % accum_steps != 0:
            continue  # accumula ancora, nessun aggiornamento dei pesi

        if max_grad_norm is not None or grad_norms is not None:
            # I gradienti sono ancora moltiplicati per la scala dell'AMP: prima si riportano
            # al valore vero, poi se la loro norma supera max_grad_norm si riducono in proporzione.
            # clip_grad_norm_ restituisce la norma PRIMA del taglio (con soglia inf non taglia nulla).
            scaler.unscale_(optimizer)
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
    """Loss media, mIoU, F1 medio, pixel accuracy e IoU per classe sull'intero loader.

    dice_weight: peso del Dice nella loss (0 = sola cross-entropy).
    desc: se passato, mostra una progress bar tqdm con questa descrizione.
    """
    model.eval()  # spegne il dropout
    cm = torch.zeros((conf.N_CLASSES, conf.N_CLASSES), dtype=torch.long, device="cuda")
    loss_sum, n_pixels = 0.0, 0
    for imgs, masks in tqdm(loader, desc=desc, disable=desc is None, dynamic_ncols=True, leave=False):
        imgs, masks = imgs.cuda(), masks.cuda().long()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(imgs)
        # La CE del batch è una media sui suoi pixel validi: la ripesiamo per il loro numero,
        # così ogni pixel conta uguale (anche quelli dell'ultimo batch, più piccolo).
        # Con il Dice è un'approssimazione (il Dice non è una media per pixel): va bene per le curve,
        # il confronto tra run si fa sul mIoU.
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
        "confusion_matrix": cm.tolist(),  # cm[vera][predetta], conteggi di pixel
    }
