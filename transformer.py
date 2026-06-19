import torch as th
import torch.nn as nn

from decoder import TransformerDecoder
from encoder import TransformerEncoder


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_enc_layers: int,
        num_dec_layers: int,
        vocab_size: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.src_embd = nn.Embedding(vocab_size, d_model)
        self.tgt_embd = nn.Embedding(vocab_size, d_model)

        self.src_pos = nn.Embedding(max_seq_len, d_model)
        self.tgt_pos = nn.Embedding(max_seq_len, d_model)

        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_enc_layers)
        self.decoder = TransformerDecoder(d_model, num_heads, d_ff, num_dec_layers)

        self.out_proj = nn.Linear(d_model, vocab_size)

        self.max_seq_len = max_seq_len

    def forward(
        self,
        src: th.Tensor,
        tgt: th.Tensor,
        src_mask: th.Tensor | None = None,
        tgt_mask: th.Tensor | None = None,
    ) -> th.Tensor:
        L_src = src.shape[1]
        L_tgt = tgt.shape[1]

        src_embd = self.src_embd(src) + self.src_pos(th.arange(L_src, device=src.device))
        tgt_embd = self.tgt_embd(tgt) + self.tgt_pos(th.arange(L_tgt, device=tgt.device))

        enc_output = self.encoder(src_embd, src_mask)
        dec_output = self.decoder(tgt_embd, enc_output, tgt_mask)

        logits = self.out_proj(dec_output)
        return logits
