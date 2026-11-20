"""Un'epoca di training e la valutazione, con mixed precision (AMP) su GPU."""
import math

import numpy as np
import torch

import conf
from metrics import confusion_matrix, mean_iou, pixel_accuracy


def train_one_epoch(model, loader, optimizer, scaler, criterion, max_steps=None, ref_loss=None):
    """Restituisce (loss di ogni iterazione, esplosa?).

    Si ferma se la loss diventa NaN o supera 3 volte la loss iniziale ref_loss (slide 9).
    """
    model.train()
    losses = []
    for step, (imgs, masks) in enumerate(loader):
        if step == max_steps:
            break
        imgs, masks = imgs.cuda(), masks.cuda().long()  # CrossEntropy vuole target int64

        with torch.autocast("cuda", dtype=torch.float16):
            loss = criterion(model(imgs), masks)

        losses.append(loss.item())
        ref_loss = ref_loss or losses[0]
        if not math.isfinite(losses[-1]) or losses[-1] > 3 * ref_loss:
            return losses, True

        optimizer.zero_grad()
        scaler.scale(loss).backward()  # loss scalata per evitare underflow dei gradienti in fp16
        scaler.step(optimizer)
        scaler.update()
    return losses, False


@torch.no_grad()
def evaluate(model, loader, criterion) -> dict[str, float]:
    """Loss media, mIoU e pixel accuracy sull'intero loader."""
    model.eval()  # spegne il dropout
    cm = torch.zeros((conf.N_CLASSES, conf.N_CLASSES), dtype=torch.long, device="cuda")
    losses = []
    for imgs, masks in loader:
        imgs, masks = imgs.cuda(), masks.cuda().long()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(imgs)
        losses.append(criterion(logits.float(), masks).item())
        cm += confusion_matrix(logits.argmax(dim=1), masks, conf.N_CLASSES, conf.IGNORE_INDEX)
    return {"loss": float(np.mean(losses)), "miou": mean_iou(cm), "pixel_acc": pixel_accuracy(cm)}
