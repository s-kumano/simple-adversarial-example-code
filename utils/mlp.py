import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Proj(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        matrix = torch.empty(out_dim, in_dim)
        self.register_buffer('matrix', matrix)
        self.matrix: Tensor

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.matrix)


class Block(nn.Module):
    def __init__(self, dim: int, residual: bool) -> None:
        super().__init__()
        self.residual = residual
        self.linear = nn.Linear(dim, dim)
        self.act = nn.ReLU(True)
        if residual:
            self.proj = Proj(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        out = self.act(out)
        if self.residual:
            out = self.proj(out)
            out += x
        return out


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layer: int, 
        residual: bool,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layer = n_layer
        self.residual = residual

        self.flatten = nn.Flatten()
        self.in_proj = Proj(in_dim, hidden_dim)
        self.blocks = nn.Sequential(*[Block(self.hidden_dim, self.residual) for _ in range(n_layer)])
        self.out_proj = Proj(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        x = self.in_proj(x)
        x = self.blocks(x)
        x = self.out_proj(x)
        return x
