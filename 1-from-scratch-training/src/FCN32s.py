import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
import torch


from VGGBackbone import VGGBackbone
from FCNHead32s import FCNHead32s
from utils import get_output_shape
import conf

class FCN32s(nn.Module):
    """Rete completa: backbone + head, collegate in forward."""
    def __init__(self, n_class=conf.N_CLASSES):
        super().__init__()
        self.backbone = VGGBackbone()
        self.head = FCNHead32s(n_class)

    def forward(self, x):
        h = self.head(self.backbone(x))
        # allinea la dimensione di uscita a quella dell'input
        return h[:, :, :x.size(2), :x.size(3)].contiguous()

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

def main():
    model = FCN32s()
    model.print_shapes()

    x = torch.zeros(conf.INPUT_SIZE_CF)
    print(model(x).shape)

if(__name__ == "__main__"):
    main()


