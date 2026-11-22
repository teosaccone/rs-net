import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

import conf
from FCN32s import check_dataset_and_forward
from FCNHead32s import FCNHead32s
from utils import get_bilinear_kernel
from VGGBackbone import VGGBackbone


class FCN8s(nn.Module):
    """FCN-8s: come la 16s, più un secondo skip da pool3 che aggiunge dettagli a stride 8.

    pool5 -> fc6 -> fc7 -> score_fr (stride 32) -> upscore2 (×2) ─┐
    pool4 ──────────────────────────────> score_pool4 (stride 16) ─+-> upscore_pool4 (×2) ─┐
    pool3 ──────────────────────────────────────────────────> score_pool3 (stride 8) ─────+-> upscore8 (×8) -> output

    Rispetto al paper: niente scala fissa su pool3 (×0.0001) e pool4 (×0.01). Gli skip partono da
    zero e imparano alla velocità normale, come nella nostra 16s.
    """
    def __init__(self, n_class=conf.N_CLASSES, pretrained: bool = True):
        super().__init__()
        self.backbone = VGGBackbone()
        self.head = FCNHead32s(n_class, upsample=False)  # stessi fc6/fc7/score_fr della 32s, senza ×32

        # Punteggi per classe calcolati da pool4 (512 canali) e da pool3 (256 canali)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        # ×2: da score_fr (es. 16×32) alla risoluzione di pool4 (32×64). Allenabile (stesso nome della 16s).
        self.upscore2 = nn.ConvTranspose2d(n_class, n_class, kernel_size=4, stride=2, padding=1, bias=False)
        # ×2: dalla risoluzione di pool4 (32×64) a quella di pool3 (64×128). Allenabile.
        self.upscore_pool4 = nn.ConvTranspose2d(n_class, n_class, kernel_size=4, stride=2, padding=1, bias=False)
        # ×8: dalla risoluzione di pool3 a quella dell'immagine. Congelato.
        # Dimensione: (64 - 1) * 8 - 2 * 4 + 16 = 512
        self.upscore8 = nn.ConvTranspose2d(n_class, n_class, kernel_size=16, stride=8, padding=4, bias=False)

        self._initialize_weights()
        if pretrained:
            self.load_pretrained_vgg16()

    def forward(self, x):
        pool3, pool4, pool5 = self.backbone(x)
        # Primo skip (come nella 16s): uscita "da 32s" ingrandita ×2 + punteggi di pool4
        h = self.upscore2(self.head(pool5))
        p4 = self.score_pool4(pool4)
        # Con dimensioni non divisibili per 32 (ceil_mode) h può essere 1 pixel più grande: si ritaglia
        h = h[:, :, :p4.size(2), :p4.size(3)] + p4
        # Secondo skip: risultato a stride 16 ingrandito ×2 + punteggi di pool3
        h = self.upscore_pool4(h)
        p3 = self.score_pool3(pool3)
        h = h[:, :, :p3.size(2), :p3.size(3)] + p3
        h = self.upscore8(h)
        # allinea la dimensione di uscita a quella dell'input
        return h[:, :, :x.size(2), :x.size(3)].contiguous()

    def _initialize_weights(self):
        """Skip a zero (all'inizio la rete è identica a una 32s), upsampling bilineari."""
        for score in (self.score_pool4, self.score_pool3):
            nn.init.zeros_(score.weight)
            nn.init.zeros_(score.bias)
        with torch.no_grad():
            for up in (self.upscore2, self.upscore_pool4, self.upscore8):
                up.weight.copy_(get_bilinear_kernel(up.in_channels, up.out_channels, up.kernel_size[0]))
        # Solo l'upsampling finale è congelato; quelli intermedi partono bilineari e possono migliorare
        self.upscore8.weight.requires_grad = False

    def load_pretrained_vgg16(self):
        """Scarica VGG16 pretrained e popola backbone e head (come in FCN32s)."""
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.backbone.load_vgg16_weights(vgg)
        self.head.copy_params_from_vgg16(vgg)


def main():
    model = FCN8s()
    x = torch.zeros(conf.INPUT_SIZE_CF)
    pool3, pool4, pool5 = model.backbone(x)
    print(f"pool3: {tuple(pool3.shape)} | pool4: {tuple(pool4.shape)} | pool5: {tuple(pool5.shape)} "
          f"| output: {tuple(model(x).shape)}")

    # Controlli sul dataset reale
    check_dataset_and_forward(model)


if __name__ == "__main__":
    main()
