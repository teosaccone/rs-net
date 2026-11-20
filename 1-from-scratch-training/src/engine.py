"""Costruzione del modello, un'epoca di training e la valutazione, con mixed precision (AMP) su GPU."""
import math

import torch
from tqdm import tqdm

import conf
from FCN32s import FCN32s
from metrics import confusion_matrix, iou_per_class, mean_f1, mean_iou, pixel_accuracy
from utils import seed_everything


def build_model(lr: float, wd: float, dropout: bool = True, momentum: float = 0.9):
    """FCN32s pre-addestrata + SGD + GradScaler, sempre con lo stesso seed (stessa inizializzazione)."""
    seed_everything(conf.SEED)
    model = FCN32s(pretrained=True).cuda()
    if not dropout:
        model.head.drop6.p = model.head.drop7.p = 0.0
    params = [p for p in model.parameters() if p.requires_grad]  # upscore è congelato
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
    return model, optimizer, torch.amp.GradScaler("cuda")


def train_one_epoch(model, loader, optimizer, scaler, criterion, max_steps=None, ref_loss=None,
                    scheduler=None, max_grad_norm=None, grad_norms=None, accum_steps=1,
                    cm=None, desc=None):
    """Restituisce (loss di ogni micro-batch, esplosa?).

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
            loss = criterion(logits, masks)

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
def evaluate(model, loader, criterion, desc=None) -> dict:
    """Loss media, mIoU, F1 medio, pixel accuracy e IoU per classe sull'intero loader.

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
        # così ogni pixel conta uguale (anche quelli dell'ultimo batch, più piccolo)
        n_valid = int((masks != conf.IGNORE_INDEX).sum())
        if n_valid > 0:
            loss_sum += criterion(logits.float(), masks).item() * n_valid
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
