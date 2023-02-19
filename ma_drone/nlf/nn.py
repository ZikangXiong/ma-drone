from typing import List

from torch import nn


class NLF(nn.Module):
    def __init__(self, arch: List[int] = None):
        super().__init__()

        if arch is None:
            arch = [12, 64, 64, 1]

        self.layers = nn.ModuleList()
        for i in range(len(arch) - 1):
            self.layers.append(nn.Linear(arch[i], arch[i + 1]))
            self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
