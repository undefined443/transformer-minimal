import torch as th
import torch.nn as nn

from self_attention import SelfAttention


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, num_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))

    def forward(self, x: th.Tensor, mask: th.Tensor | None = None) -> th.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

    def forward(self, x: th.Tensor, mask: th.Tensor | None = None) -> th.Tensor:
        for block in self.blocks:
            x = block(x, mask)
        return x
