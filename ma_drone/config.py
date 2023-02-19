import os

import torch as th

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOT_ASSETS_PATH = os.path.join(ROOT, "ma_drone/envs/assets")
DATA_ROOT = os.path.join(ROOT, "data")
DEFAULT_TENSOR_TYPE = th.float32
DEFAULT_DEVICE = "cpu"
