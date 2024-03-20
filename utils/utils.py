from typing import List

import torch
from torch import Tensor
from torch.nn import Module
from torchvision.transforms.functional import normalize


@torch.no_grad()
def in_range(x: Tensor, min: float, max: float) -> bool:
    return ((min<=x.min()) & (x.max()<=max)).item() # type: ignore


def freeze(model: Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


class ModelWithNormalization(Module):
    def __init__(self, model: Module, mean: List[float], std: List[float]) -> None:
        super().__init__()
        self.model = model
        self.mean = mean
        self.std = std

    def forward(self, x: Tensor) -> Tensor:
        assert in_range(x, 0, 1)
        return self.model(normalize(x, self.mean, self.std))