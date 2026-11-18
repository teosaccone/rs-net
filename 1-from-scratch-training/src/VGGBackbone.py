import torch.nn as nn
import torch.nn.functional as F


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


class VGGBackbone(nn.Module):
    """Backbone VGG16 senza il classificatore: restituisce pool5 (stride 32)."""
    def __init__(self):
        super().__init__()
        self.features = make_vgg_features()

    def forward(self, x):
        return self.features(x)