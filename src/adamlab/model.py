"""Decoder-only pre-LN Transformer architecture with causal self-attention."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using deterministic math SDP backend."""

    def __init__(self, width: int, n_heads: int) -> None:
        super().__init__()
        assert width % n_heads == 0, f"width {width} must be divisible by n_heads {n_heads}"
        self.width = width
        self.n_heads = n_heads
        self.head_dim = width // n_heads

        self.qkv = nn.Linear(width, 3 * width, bias=True)
        self.proj = nn.Linear(width, width, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Enforce deterministic MATH SDP backend
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        res: torch.Tensor = self.proj(out)
        return res


class MLP(nn.Module):
    """Transformer MLP block with 4x expansion and GELU activation."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, 4 * width, bias=True)
        self.fc2 = nn.Linear(4 * width, width, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res: torch.Tensor = self.fc2(F.gelu(self.fc1(x), approximate="none"))
        return res


class TransformerBlock(nn.Module):
    """Pre-LN Transformer layer block."""

    def __init__(self, width: int, n_heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = CausalSelfAttention(width, n_heads)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = MLP(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only pre-LN language model."""

    def __init__(
        self,
        vocab_size: int = 65,
        width: int = 512,
        depth: int = 4,
        context_length: int = 128,
        init_generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.width = width
        self.depth = depth
        self.context_length = context_length

        n_heads = width // 64
        assert n_heads * 64 == width, f"Head dimension must be 64, got width={width}"

        self.tok_emb = nn.Embedding(vocab_size, width)
        self.pos_emb = nn.Embedding(context_length, width)

        self.blocks = nn.ModuleList([TransformerBlock(width, n_heads) for _ in range(depth)])

        self.ln_f = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)

        self._init_weights(init_generator)

    def _init_weights(self, gen: torch.Generator | None) -> None:
        """Initialize weights with custom standard deviation per specification."""
        std_base = 0.02
        std_residual = 0.02 / math.sqrt(2 * self.depth)

        with torch.no_grad():
            # Helper to draw normal values with generator
            def init_tensor(tensor: torch.Tensor, mean: float, std: float) -> None:
                if gen is not None:
                    # torch.randn with generator
                    rand_vals = torch.randn(
                        tensor.shape, generator=gen, dtype=tensor.dtype, device=tensor.device
                    )
                    tensor.data.copy_(rand_vals.mul_(std).add_(mean))
                else:
                    nn.init.normal_(tensor, mean=mean, std=std)

        # Token & Positional embeddings
        init_tensor(self.tok_emb.weight, 0.0, std_base)
        init_tensor(self.pos_emb.weight, 0.0, std_base)

        for blk in self.blocks:
            assert isinstance(blk, TransformerBlock)
            block = blk
            # LayerNorm 1
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)

            # Attention QKV
            init_tensor(block.attn.qkv.weight, 0.0, std_base)
            nn.init.zeros_(block.attn.qkv.bias)

            # Attention Projection (residual branch)
            init_tensor(block.attn.proj.weight, 0.0, std_residual)
            nn.init.zeros_(block.attn.proj.bias)

            # LayerNorm 2
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)

            # MLP FC1
            init_tensor(block.mlp.fc1.weight, 0.0, std_base)
            nn.init.zeros_(block.mlp.fc1.bias)

            # MLP FC2 (residual branch)
            init_tensor(block.mlp.fc2.weight, 0.0, std_residual)
            nn.init.zeros_(block.mlp.fc2.bias)

        # Final LayerNorm
        nn.init.ones_(self.ln_f.weight)
        nn.init.zeros_(self.ln_f.bias)

        # Output LM Head
        init_tensor(self.head.weight, 0.0, std_base)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.context_length, (
            f"Cannot forward sequence of length {T}, context is {self.context_length}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits: torch.Tensor = self.head(x)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
