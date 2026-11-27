import torch.nn as nn
import torch.nn.functional as F
import torch

import conf
from utils import get_bilinear_kernel

class FCNHead32s(nn.Module):
    """Parte fully convolutional: fc6/fc7 convertiti in conv, score e (opzionale) upsampling ×32.

    upsample=False: si ferma a score_fr (mappa a stride 32) e non crea upscore.
    Serve a FCN16s, che fa l'upsampling a modo suo dopo aver sommato lo skip da pool4.
    """
    def __init__(self, n_class=conf.N_CLASSES, upsample: bool = True):
        super().__init__()
        self.fc6 = nn.Conv2d(512, 4096, 7, padding=3)
        self.drop6 = nn.Dropout2d(0.5)
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.drop7 = nn.Dropout2d(0.5)

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.upscore = None
        if upsample:
            self.upscore = nn.ConvTranspose2d(n_class, n_class, kernel_size=64,
                                              stride=32, bias=False, padding=16)

        # Init di score_fr (zeri) e upscore (bilineare congelato). Va fatta qui:
        # copy_params_from_vgg16 tocca solo fc6/fc7, quindi non la sovrascrive.
        self._initialize_weights()

    def forward(self, x):
        h = self.drop6(F.relu(self.fc6(x)))
        h = self.drop7(F.relu(self.fc7(h)))
        h = self.score_fr(h)
        if self.upscore is None:
            return h
        return self.upscore(h)

    def init_random_weights(self):
        """fc6 e fc7 casuali (init di He fan_out, come il backbone) per il training da zero.

        Da noi fc6/fc7 sono convoluzioni, quindi ricevono la stessa init delle conv della VGG di
        torchvision (che invece, come layer Linear, li inizializza con N(0, 0.01)).
        score_fr resta a zero e upscore bilineare: non vengono da ImageNet, sono uguali nei due casi.
        """
        for conv in (self.fc6, self.fc7):
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(conv.bias)

    def copy_params_from_vgg16(self, vgg16_model: nn.Module):
        """Reshapa e copia i pesi dei layer FC (fc6, fc7) di VGG16 nei Conv2d."""
        fc_mapping = [(0, self.fc6), (3, self.fc7)]
        with torch.no_grad():
            for idx, conv_layer in fc_mapping:
                linear_layer = vgg16_model.classifier[idx]
                conv_layer.weight.copy_(linear_layer.weight.view_as(conv_layer.weight))
                conv_layer.bias.copy_(linear_layer.bias.view_as(conv_layer.bias))

    def _initialize_weights(self):
        """Inizializza score_fr a zero e upscore con filtro bilineare (Long et al.)."""
        nn.init.zeros_(self.score_fr.weight)
        if self.score_fr.bias is not None:
            nn.init.zeros_(self.score_fr.bias)

        if self.upscore is None:
            return

        bilinear_weight = get_bilinear_kernel(
            self.upscore.in_channels,
            self.upscore.out_channels,
            self.upscore.kernel_size[0],
        )
        with torch.no_grad():
            self.upscore.weight.copy_(bilinear_weight)

        # Congelo i pesi di upsampling bilineare
        self.upscore.weight.requires_grad = False
