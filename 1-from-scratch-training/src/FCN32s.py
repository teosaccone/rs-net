import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

import conf
from dataset import build_dataloaders
from FCNHead32s import FCNHead32s
from utils import decode_segmap, get_output_shape
from VGGBackbone import VGGBackbone


def visualize_sample(img_tensor: torch.Tensor, mask_tensor: torch.Tensor):
    """Denormalizza un'immagine [3, H, W] e la mostra affiancata alla ground truth [H, W]."""
    mean = np.array(conf.IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(conf.IMAGENET_STD).reshape(3, 1, 1)

    img_np = img_tensor.detach().cpu().numpy() * std + mean
    img_np = np.clip(img_np.transpose(1, 2, 0), 0.0, 1.0)

    mask_rgb = decode_segmap(mask_tensor.detach().cpu().numpy())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(img_np)
    axes[0].set_title("Immagine RGB (Denormalizzata)")
    axes[0].axis("off")

    axes[1].imshow(mask_rgb)
    axes[1].set_title("Ground Truth (19 classi + ignore nero)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

class FCN32s(nn.Module):
    """Rete completa: backbone + head, collegate in forward."""
    def __init__(self, n_class=conf.N_CLASSES, pretrained: bool = True):
        super().__init__()
        self.backbone = VGGBackbone()
        self.head = FCNHead32s(n_class)

        if pretrained:
            self.load_pretrained_vgg16()

    def forward(self, x):
        _, pool5 = self.backbone(x)  # la 32s usa solo pool5 (pool4 serve alla 16s)
        h = self.head(pool5)
        # allinea la dimensione di uscita a quella dell'input
        return h[:, :, :x.size(2), :x.size(3)].contiguous()

    def load_pretrained_vgg16(self):
        """Scarica VGG16 pretrained e popola backbone e head."""
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.backbone.load_vgg16_weights(vgg)
        self.head.copy_params_from_vgg16(vgg)

    def print_shapes(self, input_size=conf.INPUT_SIZE_CF):
        """
        Stampa la shape in output di ogni layer della rete mandando un tensore lungo
        tutta la rete e mandandone in output le dimensioni.
        """
        print(f"{'Layer':<40} {'Output shape'}")
        print("-" * 60)

        current_size = input_size
        for name, layer in self.named_modules():
            if len(list(layer.children())) > 0:  # salta i moduli contenitore
                continue
            current_size = get_output_shape(current_size, layer)
            print(f"{name:<40} {current_size}")



def check_dataset_and_forward(model: nn.Module):
    """Sanity check end-to-end: DataLoader -> Maschere -> Forward -> Loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[CHECK] Device rilevato: {device}")

    # 1. Costruzione DataLoader
    train_dl, val_dl, test_dl = build_dataloaders()
    print(f"[CHECK] Batch count -> Train: {len(train_dl)}, Val: {len(val_dl)}, Test: {len(test_dl)}")

    # 2. Estrazione primo batch
    imgs, masks = next(iter(train_dl))
    # Visualizza la prima immagine del batch e la relativa maschera
    visualize_sample(imgs[0], masks[0])

    print(f"[CHECK] Imgs shape: {imgs.shape} | dtype: {imgs.dtype} | range: [{imgs.min():.2f}, {imgs.max():.2f}]")
    print(f"[CHECK] Masks shape: {masks.shape} | dtype: {masks.dtype}")

    # 3. Controllo classi valide nelle maschere (0..N_CLASSES-1 oppure IGNORE_INDEX)
    unique_labels = torch.unique(masks).tolist()
    valid_labels = set(range(conf.N_CLASSES)) | {conf.IGNORE_INDEX}
    invalid = set(unique_labels) - valid_labels
    assert not invalid, f"Trovate label non valide nella maschera: {invalid}"
    print(f"[CHECK] Classi presenti nel batch: {unique_labels}")

    # 4. Forward pass e calcolo loss su GPU/CPU
    model = model.to(device).eval()
    imgs = imgs.to(device)
    masks = masks.to(device).long()

    with torch.no_grad():
        logits = model(imgs)
        assert logits.shape == (imgs.size(0), conf.N_CLASSES, imgs.size(2), imgs.size(3)), (
            f"Mismatch output shape: atteso {(imgs.size(0), conf.N_CLASSES, imgs.size(2), imgs.size(3))}, ottenuto {logits.shape}"
        )
        criterion = nn.CrossEntropyLoss(ignore_index=conf.IGNORE_INDEX)
        loss = criterion(logits, masks)

    print(f"[CHECK] Logits shape: {logits.shape} | Initial Loss: {loss.item():.4f}")
    if device.type == "cuda":
        vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"[CHECK] Picco VRAM (eval, batch={imgs.size(0)}): {vram_mb:.1f} MB")


def main():
    model = FCN32s()
    model.print_shapes()

    x = torch.zeros(conf.INPUT_SIZE_CF)
    print(f"\nDummy forward output shape: {model(x).shape}")

    # Controlli sul dataset reale
    check_dataset_and_forward(model)


if __name__ == "__main__":
    main()