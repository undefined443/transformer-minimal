import torch as th
import torch.nn as nn

from cross_attention import CrossAttention
from self_attention import SelfAttention


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        self.self_attn = SelfAttention(d_model, num_heads)
        self.cross_attn = CrossAttention(d_model, num_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self, x: th.Tensor, encoder_output: th.Tensor, mask: th.Tensor | None = None
    ) -> th.Tensor:
        x = x + self.self_attn(self.norm1(x), mask)
        x = x + self.cross_attn(self.norm2(x), encoder_output, encoder_output)
        x = x + self.ffn(self.norm3(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerDecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

    def forward(
        self, x: th.Tensor, encoder_output: th.Tensor, mask: th.Tensor | None = None
    ) -> th.Tensor:
        for block in self.blocks:
            x = block(x, encoder_output, mask)
        return x
