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


class VGGBackbone(nn.Module):
    """Backbone VGG16 senza il classificatore: restituisce pool5 (stride 32)."""
    def __init__(self):
        super().__init__()
        self.features = make_vgg_features()

    def forward(self, x):
        return self.features(x)

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