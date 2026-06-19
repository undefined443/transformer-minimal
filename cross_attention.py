import torch as th
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self, x: th.Tensor, k: th.Tensor, v: th.Tensor, mask: th.Tensor | None = None
    ) -> th.Tensor:
        B, L, D = x.shape
        H, HD = self.num_heads, self.head_dim

        def split_heads(z: th.Tensor) -> th.Tensor:
            L = z.shape[1]
            z = z.view(B, L, H, HD).transpose(1, 2)
            return z

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(k))
        v = split_heads(self.v_proj(v))

        scale = HD**0.5
        attn = q @ k.transpose(-2, -1) / scale

        if mask is not None:
            attn += mask * -1e9

        out = attn.softmax(dim=-1) @ v
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)
