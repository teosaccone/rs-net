import torch
import torch.nn as nn


VGG16_CFG = [64, 64, 'M', 
             128, 128, 'M',
             256, 256, 256, 'M', 
             512, 512, 512, 'M',
             512, 512, 512, 'M']


def make_vgg_features(cfg=VGG16_CFG):
    """Questa funzione genera la backbone di VGG sfruttando la sola lista di configurazione"""
    layers, cin = [], 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))
        else:
            layers += [nn.Conv2d(cin, v, 3, padding=1), nn.ReLU(inplace=True)]
            cin = v
    return nn.Sequential(*layers)


# Indici in self.features subito dopo il 3° e il 4° maxpool: features[:17] finisce con pool3,
# features[:24] con pool4. (ogni blocco è conv+ReLU ripetuti + un maxpool; il 3° maxpool è
# all'indice 16, il 4° all'indice 23)
POOL3_END = 17
POOL4_END = 24


class VGGBackbone(nn.Module):
    """Backbone VGG16 senza il classificatore: restituisce pool3 (stride 8), pool4 (stride 16) e pool5 (stride 32)."""
    def __init__(self):
        super().__init__()
        self.features = make_vgg_features()

    def forward(self, x):
        # Stessi layer di sempre, eseguiti in tre pezzi per leggere anche i risultati intermedi.
        # self.features resta un unico Sequential: i nomi dei pesi (features.N) non cambiano.
        pool3 = self.features[:POOL3_END](x)
        pool4 = self.features[POOL3_END:POOL4_END](pool3)
        pool5 = self.features[POOL4_END:](pool4)
        return pool3, pool4, pool5

    def init_random_weights(self):
        """Pesi casuali per il training da zero (senza ImageNet): init di He (Kaiming), bias a zero.

        Senza BatchNorm, con 13 conv + ReLU di fila, l'init di default di PyTorch fa rimpicciolire il
        segnale a ogni strato. Con std = sqrt(2 / fan_in) la varianza delle attivazioni resta circa
        costante attraverso le ReLU (He et al. 2015): il segnale arriva vivo fino a pool5.
        """
        for m in self.features:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def load_vgg16_weights(self, vgg16_model: nn.Module):
        """Copia i pesi convoluzionali da vgg16.features."""
        src_convs = [m for m in vgg16_model.features if isinstance(m, nn.Conv2d)]
        dst_convs = [m for m in self.features if isinstance(m, nn.Conv2d)]
        
        assert len(src_convs) == len(dst_convs), "Numero di layer Conv2d non corrispondente"
        with torch.no_grad():
            for src, dst in zip(src_convs, dst_convs):
                assert src.weight.shape == dst.weight.shape
                dst.weight.copy_(src.weight)
                dst.bias.copy_(src.bias)