"""Reti FCN-32s/16s/8s su VGG16: copia di VGGBackbone.py, FCNHead32s.py, FCN32s.py, FCN16s.py, FCN8s.py del punto 1.

Differenze rispetto al punto 1:
- tutte in un file, senza dipendenze dal conf di Cityscapes;
- n_class obbligatorio (19 per caricare la rete di Cityscapes, 11 per CamVid);
- tolte le funzioni di controllo legate al dataset Cityscapes (check_dataset_and_forward, print_shapes).
I nomi dei layer sono IDENTICI: i best.pth del punto 1 si caricano così come sono
(i nomi dei pesi dipendono dagli attributi, es. backbone.features.0.weight, non dal file).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16

from camvid_utils import get_bilinear_kernel

VGG16_CFG = [64, 64, "M",
             128, 128, "M",
             256, 256, 256, "M",
             512, 512, 512, "M",
             512, 512, 512, "M"]
# Indici in features subito dopo il 3° e il 4° maxpool (features[:17] finisce con pool3, features[:24] con pool4)
POOL3_END = 17
POOL4_END = 24


def make_vgg_features(cfg: list = VGG16_CFG) -> nn.Sequential:
    """Backbone di VGG generata dalla lista di configurazione (numero = conv 3x3 + ReLU, "M" = maxpool)."""
    layers, cin = [], 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))
        else:
            layers += [nn.Conv2d(cin, v, 3, padding=1), nn.ReLU(inplace=True)]
            cin = v
    return nn.Sequential(*layers)


class VGGBackbone(nn.Module):
    """VGG16 senza classificatore: restituisce pool3 (stride 8), pool4 (stride 16) e pool5 (stride 32)."""

    def __init__(self):
        super().__init__()
        self.features = make_vgg_features()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pool3 = self.features[:POOL3_END](x)
        pool4 = self.features[POOL3_END:POOL4_END](pool3)
        pool5 = self.features[POOL4_END:](pool4)
        return pool3, pool4, pool5

    def init_random_weights(self) -> None:
        """Init di He (fan_out, come vgg16(weights=None) di torchvision), bias a zero."""
        for m in self.features:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def load_vgg16_weights(self, vgg16_model: nn.Module) -> None:
        """Copia i pesi convoluzionali da vgg16.features."""
        src_convs = [m for m in vgg16_model.features if isinstance(m, nn.Conv2d)]
        dst_convs = [m for m in self.features if isinstance(m, nn.Conv2d)]
        assert len(src_convs) == len(dst_convs), "Numero di layer Conv2d non corrispondente"
        with torch.no_grad():
            for src, dst in zip(src_convs, dst_convs):
                dst.weight.copy_(src.weight)
                dst.bias.copy_(src.bias)


class FCNHead32s(nn.Module):
    """fc6/fc7 convertiti in conv, score_fr (a zero) e, se upsample, upscore x32 bilineare congelato."""

    def __init__(self, n_class: int, upsample: bool = True):
        super().__init__()
        self.fc6 = nn.Conv2d(512, 4096, 7, padding=3)
        self.drop6 = nn.Dropout2d(0.5)
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.drop7 = nn.Dropout2d(0.5)
        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.upscore = None
        if upsample:
            self.upscore = nn.ConvTranspose2d(n_class, n_class, kernel_size=64, stride=32, bias=False, padding=16)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop6(F.relu(self.fc6(x)))
        h = self.drop7(F.relu(self.fc7(h)))
        h = self.score_fr(h)
        return h if self.upscore is None else self.upscore(h)

    def init_random_weights(self) -> None:
        """fc6 e fc7 casuali (He fan_out); score_fr resta a zero."""
        for conv in (self.fc6, self.fc7):
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(conv.bias)

    def copy_params_from_vgg16(self, vgg16_model: nn.Module) -> None:
        """Reshape e copia dei layer fully connected fc6, fc7 di VGG16 nelle conv."""
        with torch.no_grad():
            for idx, conv in [(0, self.fc6), (3, self.fc7)]:
                linear = vgg16_model.classifier[idx]
                conv.weight.copy_(linear.weight.view_as(conv.weight))
                conv.bias.copy_(linear.bias.view_as(conv.bias))

    def _initialize_weights(self) -> None:
        nn.init.zeros_(self.score_fr.weight)
        nn.init.zeros_(self.score_fr.bias)
        if self.upscore is not None:
            with torch.no_grad():
                self.upscore.weight.copy_(get_bilinear_kernel(self.upscore.in_channels, self.upscore.out_channels,
                                                              self.upscore.kernel_size[0]))
            self.upscore.weight.requires_grad = False


def _load_imagenet_or_random(backbone: VGGBackbone, head: FCNHead32s, pretrained: bool) -> None:
    """pretrained: pesi VGG16 ImageNet; altrimenti init di He (come nel punto 1)."""
    if pretrained:
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        backbone.load_vgg16_weights(vgg)
        head.copy_params_from_vgg16(vgg)
    else:
        backbone.init_random_weights()
        head.init_random_weights()


class FCN32s(nn.Module):
    """FCN-32s: backbone + head con upsampling x32."""

    def __init__(self, n_class: int, pretrained: bool = True):
        super().__init__()
        self.backbone = VGGBackbone()
        self.head = FCNHead32s(n_class)
        _load_imagenet_or_random(self.backbone, self.head, pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, pool5 = self.backbone(x)
        h = self.head(pool5)
        return h[:, :, :x.size(2), :x.size(3)].contiguous()


class FCN16s(nn.Module):
    """FCN-16s: 32s + skip da pool4 (score_pool4 a zero, upscore2 x2 allenabile, upscore16 congelato)."""

    def __init__(self, n_class: int, pretrained: bool = True):
        super().__init__()
        self.backbone = VGGBackbone()
        self.head = FCNHead32s(n_class, upsample=False)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)
        self.upscore2 = nn.ConvTranspose2d(n_class, n_class, kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore16 = nn.ConvTranspose2d(n_class, n_class, kernel_size=32, stride=16, padding=8, bias=False)
        nn.init.zeros_(self.score_pool4.weight)
        nn.init.zeros_(self.score_pool4.bias)
        with torch.no_grad():
            for up in (self.upscore2, self.upscore16):
                up.weight.copy_(get_bilinear_kernel(up.in_channels, up.out_channels, up.kernel_size[0]))
        self.upscore16.weight.requires_grad = False
        _load_imagenet_or_random(self.backbone, self.head, pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, pool4, pool5 = self.backbone(x)
        h = self.upscore2(self.head(pool5))
        p4 = self.score_pool4(pool4)
        h = h[:, :, :p4.size(2), :p4.size(3)] + p4  # ritaglio di 1 pixel se le dimensioni non dividono 32
        h = self.upscore16(h)
        return h[:, :, :x.size(2), :x.size(3)].contiguous()


class FCN8s(nn.Module):
    """FCN-8s: 16s + skip da pool3 (score_pool3 a zero, upscore_pool4 x2 allenabile, upscore8 congelato).

    Come nel punto 1: niente scala fissa 0.0001 / 0.01 del paper sugli skip.
    """

    def __init__(self, n_class: int, pretrained: bool = True):
        super().__init__()
        self.backbone = VGGBackbone()
        self.head = FCNHead32s(n_class, upsample=False)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.upscore2 = nn.ConvTranspose2d(n_class, n_class, kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(n_class, n_class, kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore8 = nn.ConvTranspose2d(n_class, n_class, kernel_size=16, stride=8, padding=4, bias=False)
        for score in (self.score_pool4, self.score_pool3):
            nn.init.zeros_(score.weight)
            nn.init.zeros_(score.bias)
        with torch.no_grad():
            for up in (self.upscore2, self.upscore_pool4, self.upscore8):
                up.weight.copy_(get_bilinear_kernel(up.in_channels, up.out_channels, up.kernel_size[0]))
        self.upscore8.weight.requires_grad = False
        _load_imagenet_or_random(self.backbone, self.head, pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pool3, pool4, pool5 = self.backbone(x)
        h = self.upscore2(self.head(pool5))
        p4 = self.score_pool4(pool4)
        h = h[:, :, :p4.size(2), :p4.size(3)] + p4
        h = self.upscore_pool4(h)
        p3 = self.score_pool3(pool3)
        h = h[:, :, :p3.size(2), :p3.size(3)] + p3
        h = self.upscore8(h)
        return h[:, :, :x.size(2), :x.size(3)].contiguous()


# Nome (come in camvid_conf.MODEL) -> classe della rete
MODELS = {"fcn32s": FCN32s, "fcn16s": FCN16s, "fcn8s": FCN8s}
