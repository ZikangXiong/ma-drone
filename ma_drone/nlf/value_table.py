from typing import List, Union

import numpy as np
import torch as th

from ma_drone.config import DATA_ROOT
from ma_drone.nlf.nn import NLF
from ma_drone.utils import default_tensor


class ValueTable:
    def __init__(self):
        self.table = {}
        all_obs = np.load(f"{DATA_ROOT}/transitions.npy").reshape(-1, 12)
        self.upper = all_obs.max(axis=0)
        self.lower = all_obs.min(axis=0)
        self.lr = 1e-1
        self.pgd_steps = 400

    def precompute_table_with_nlf(self, nlf: NLF,
                                  radius_list: Union[List[float], np.ndarray],
                                  n_samples: int,
                                  constraint_coeff: float = 100):
        # difference to previous implementation:
        # fixing r search for v, instead of fixing v search for r
        for r in radius_list:
            v = self._pgd_with_nlf(nlf, r, n_samples, constraint_coeff)
            self.table[v] = r

    def _pgd_with_nlf(self, nlf: NLF, radius: float, n_samples: int, constraint_coeff: float = 100):
        x = np.random.uniform(self.lower, self.upper, size=(n_samples, 12))
        x = default_tensor(x, requires_grad=True)
        optimizer = th.optim.Adam([x], lr=self.lr)

        v = nlf(x)
        dist_to_radius = th.abs(th.norm(x, dim=1) - radius)
        for _ in range(self.pgd_steps):
            x.data = x.data.clamp(default_tensor(self.lower), default_tensor(self.upper))
            dist_to_radius = th.abs(th.norm(x, dim=1) - radius)
            v = nlf(x)
            # find minimum v trigger this radius
            loss = v.mean() + constraint_coeff * dist_to_radius.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"v: {v.mean().item()}, dist: {dist_to_radius.mean().item()}")
        return v.mean().item()

    def save(self, path: str):
        np.save(path, self.table)

    @classmethod
    def load(cls, path: str) -> 'ValueTable':
        vt = cls()
        vt.table = np.load(path, allow_pickle=True).item()
        return vt
