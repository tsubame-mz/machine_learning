import collections
from typing import Optional
import numpy as np
import torch

MAXIMUM_FLOAT_VALUE = float("inf")
KnownBounds = collections.namedtuple("KnownBounds", ["min", "max"])


class MinMaxStats:
    """
    最大値と最小値を保持しておくためのクラス
    """

    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        if value is None:
            raise ValueError

        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if value is None:
            return 0.0
        if self.maximum > self.minimum:
            # 標準化
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

    def __repr__(self):
        return f"MinMaxStats[min[{self.minimum}]/max[{self.maximum}]]"


def softmax(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x)
    exp = np.exp(x - x_max)
    return exp / np.sum(exp)


def get_device(use_gpu):
    # サポート対象のGPUがあれば使う
    if use_gpu:
        print("Check GPU available")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Use device[{device}]")
    return device
