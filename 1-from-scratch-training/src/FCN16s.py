import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

import conf
from FCN32s import check_dataset_and_forward
from FCNHead32s import FCNHead32s
from utils import get_bilinear_kernel
from VGGBackbone import VGGBackbone


class FCN16s(nn.Module):
    """FCN-16s: come la 32s, più uno skip da pool4 che aggiunge dettagli a stride 16.

    pool5 -> fc6 -> fc7 -> score_fr (stride 32) -> upscore2 (×2) ─┐
                                                                  + -> upscore16 (×16) -> output
    pool4 ──────────────────────────────> score_pool4 (stride 16) ─┘
    """
    def __init__(self, n_class=conf.N_CLASSES, pretrained: bool = True):
        super().__init__()
        self.backbone = VGGBackbone()
        self.head = FCNHead32s(n_class, upsample=False)  # stessi fc6/fc7/score_fr della 32s, senza ×32

        # Punteggi per classe calcolati da pool4 (512 canali)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)
        # ×2: porta score_fr (es. 16×32) alla risoluzione di pool4 (32×64). Allenabile.
        self.upscore2 = nn.ConvTranspose2d(n_class, n_class, kernel_size=4, stride=2, padding=1, bias=False)
        # ×16: dalla risoluzione di pool4 a quella dell'immagine. Congelato.
        self.upscore16 = nn.ConvTranspose2d(n_class, n_class, kernel_size=32, stride=16, padding=8, bias=False)

        self._initialize_weights()
        if pretrained:
            self.load_pretrained_vgg16()

    def forward(self, x):
        _, pool4, pool5 = self.backbone(x)  # pool3 serve solo alla 8s
        h = self.upscore2(self.head(pool5))  # uscita "da 32s", ingrandita ×2
        p4 = self.score_pool4(pool4)
        # Con dimensioni non divisibili per 32 (ceil_mode) h può essere 1 pixel più grande: si ritaglia
        h = h[:, :, :p4.size(2), :p4.size(3)] + p4
        h = self.upscore16(h)
        # allinea la dimensione di uscita a quella dell'input
        return h[:, :, :x.size(2), :x.size(3)].contiguous()

    def _initialize_weights(self):
        """score_pool4 a zero (all'inizio la rete è identica a una 32s), upsampling bilineari."""
        nn.init.zeros_(self.score_pool4.weight)
        nn.init.zeros_(self.score_pool4.bias)
        with torch.no_grad():
            for up in (self.upscore2, self.upscore16):
                up.weight.copy_(get_bilinear_kernel(up.in_channels, up.out_channels, up.kernel_size[0]))
        # Solo l'upsampling finale è congelato; upscore2 parte bilineare e può migliorare (come nel paper)
        self.upscore16.weight.requires_grad = False

    def load_pretrained_vgg16(self):
        """Scarica VGG16 pretrained e popola backbone e head (come in FCN32s)."""
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.backbone.load_vgg16_weights(vgg)
        self.head.copy_params_from_vgg16(vgg)


def main():
    model = FCN16s()
    x = torch.zeros(conf.INPUT_SIZE_CF)
    _, pool4, pool5 = model.backbone(x)
    print(f"pool4: {tuple(pool4.shape)} | pool5: {tuple(pool5.shape)} | output: {tuple(model(x).shape)}")

    # Controlli sul dataset reale
    check_dataset_and_forward(model)


if __name__ == "__main__":
    main()
