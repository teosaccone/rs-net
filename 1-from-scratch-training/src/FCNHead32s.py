import torch.nn as nn
import torch.nn.functional as F

import conf

class FCNHead32s(nn.Module):
    """Parte fully convolutional: fc6/fc7 convertiti in conv, score e upsampling."""
    def __init__(self, n_class=conf.N_CLASSES):
        super().__init__()
        self.fc6 = nn.Conv2d(512, 4096, 7, padding=1)
        self.drop6 = nn.Dropout2d(0.5)
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.drop7 = nn.Dropout2d(0.5)

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.upscore = nn.ConvTranspose2d(n_class, n_class, kernel_size=64,
                                          stride=32, bias=False)

    def forward(self, x):
        h = self.drop6(F.relu(self.fc6(x)))
        h = self.drop7(F.relu(self.fc7(h)))
        h = self.score_fr(h)
        return self.upscore(h)