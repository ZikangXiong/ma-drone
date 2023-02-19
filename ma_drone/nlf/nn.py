from typing import List

import numpy as np
import torch as th
from torch import nn
from torch.utils.data import DataLoader

from ma_drone.utils import default_tensor


class NLF(nn.Module):
    def __init__(self, arch: List[int] = None):
        super().__init__()

        if arch is None:
            arch = [12, 64, 64, 1]

        self.layers = nn.ModuleList()
        for i in range(len(arch) - 1):
            self.layers.append(nn.Linear(arch[i], arch[i + 1]))
            self.layers.append(nn.Tanh())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return th.abs(x)

    def loss(self, batch, upper_cnst: float, upper_bound: float):
        x0 = batch[:, 0]
        x1 = batch[:, 1]

        mono_loss = th.mean(self.forward(x1) - self.forward(x0))
        upper_loss = th.max(self(batch).flatten() - upper_bound, default_tensor(0.0)).mean()

        loss = mono_loss + upper_cnst * upper_loss
        return loss, mono_loss, upper_loss

    def train_nlf(self,
                  data: np.ndarray,
                  n_epochs: int,
                  batch_size: int,
                  lr: float = 1e-3,
                  upper_cnst: float = 10,
                  weight_decay: float = 0.01,
                  upper_bound: float = 10.0):
        optimizer = th.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        data_tensor = th.from_numpy(data).float()
        data_loader = DataLoader(data_tensor, batch_size=batch_size, shuffle=True)

        for epoch in range(n_epochs):
            losses, mono_losses, upper_losses = [], [], []
            for batch in data_loader:
                optimizer.zero_grad()

                loss, mono_loss, upper_loss = self.loss(batch, upper_cnst, upper_bound)
                losses.append(loss.item())
                mono_losses.append(mono_loss.item())
                upper_losses.append(upper_loss.item())

                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch}: {np.mean(losses)} - {np.mean(mono_losses)} - {np.mean(upper_losses)}')

    def save(self, path: str):
        th.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, arch: List[int] = None) -> 'NLF':
        model = cls(arch)
        model.load_state_dict(th.load(path))
        return model
