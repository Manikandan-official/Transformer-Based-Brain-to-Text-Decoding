"""
model_transformer.py — Transformer BCI Decoder
================================================
Architecture matches 7th place GRU baseline but uses Transformer encoder.
Same interface as model.py (BCIDecoder):
  logits, projections, adj_lengths = model(neural, session_ids, lengths)

Architecture:
  Input (B, T, 512)
  → DayAdapter (linear per-session, Softsign)
  → PatchEmbedding (patch_size=16, stride=4)
  → Positional Encoding
  → Transformer Encoder (4-6 layers, d_model=512, n_heads=8)
  → Dropout
  → CTC head → log_softmax
  → ProjectionHead (contrastive)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataset import N_PHONEMES, BLANK_IDX


# ── Day Adapter (same as Mamba model) ────────────────────────────────────────
class DayAdapter(nn.Module):
    """Linear per-session adaptation with Softsign."""
    def __init__(self, n_sessions, input_dim=512):
        super().__init__()
        self.weights = nn.ParameterList([
            nn.Parameter(torch.eye(input_dim)) for _ in range(n_sessions)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(1, input_dim)) for _ in range(n_sessions)
        ])
        self.act = nn.Softsign()

    def forward(self, x, session_ids):
        out = torch.zeros_like(x)
        for sid in session_ids.unique():
            mask = (session_ids == sid)
            w = self.weights[sid.item()]
            b = self.biases[sid.item()]
            out[mask] = self.act(
                torch.einsum('btd,dk->btk', x[mask], w) + b
            ).to(x.dtype)
        return out


# ── Patch Embedding ───────────────────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    """Same patch embedding as Mamba model for fair comparison."""
    def __init__(self, input_dim, d_model, patch_size=16, stride=4):
        super().__init__()
        self.patch_size = patch_size
        self.stride     = stride
        self.proj = nn.Sequential(
            nn.Linear(input_dim * patch_size, 4096),
            nn.Softsign(),
            nn.Dropout(0.3),
            nn.Linear(4096, d_model),
        )

    def forward(self, x):
        B, T, C = x.shape
        x = x.unsqueeze(1).permute(0, 3, 1, 2)   # (B, C, 1, T)
        x = x.unfold(3, self.patch_size, self.stride)  # (B, C, 1, T', p)
        x = x.squeeze(2).permute(0, 2, 3, 1)      # (B, T', p, C)
        x = x.reshape(x.shape[0], x.shape[1], -1) # (B, T', p*C)
        return self.proj(x)

    def output_lengths(self, input_lengths):
        return ((input_lengths.float() - self.patch_size) / self.stride + 1).long().clamp(min=1)


# ── Positional Encoding ───────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ── Transformer Encoder Block ─────────────────────────────────────────────────
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff         = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Feedforward with residual
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


# ── Projection Head (contrastive) ─────────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, d_model, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )

    def forward(self, x, lengths):
        mask = (torch.arange(x.shape[1], device=x.device).unsqueeze(0)
                < lengths.unsqueeze(1)).unsqueeze(-1).float()
        pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        return F.normalize(self.net(pooled), dim=-1)


# ── Full Transformer BCI Decoder ──────────────────────────────────────────────
class BCITransformer(nn.Module):
    """
    Transformer-based BCI decoder.
    Same interface as BCIDecoder (Mamba) for easy ensemble.

    Returns: logits (T, B, vocab), projections, adj_lengths
    """
    def __init__(self,
                 n_sessions,
                 input_dim      = 512,
                 n_units        = 512,
                 n_layers       = 4,
                 n_heads        = 8,
                 dim_feedforward = 2048,
                 patch_size     = 16,
                 patch_stride   = 4,
                 dropout        = 0.4,
                 proj_dim       = 128):
        super().__init__()

        self.day_adapter = DayAdapter(n_sessions, input_dim)
        self.patch_embed = PatchEmbedding(input_dim, n_units, patch_size, patch_stride)
        self.pos_enc     = PositionalEncoding(n_units, dropout=dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=n_units,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.dropout   = nn.Dropout(dropout)
        self.ctc_head  = nn.Linear(n_units, N_PHONEMES)
        self.proj_head = ProjectionHead(n_units, proj_dim)

        nn.init.xavier_uniform_(self.ctc_head.weight)

    def forward(self, x, session_ids, lengths):
        # Day adaptation
        x = self.day_adapter(x, session_ids)

        # Patch embedding
        x = self.patch_embed(x)
        adj_lengths = self.patch_embed.output_lengths(lengths)

        # Positional encoding
        x = self.pos_enc(x)

        # Build key_padding_mask for variable-length sequences
        B, T, _ = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) >= adj_lengths.unsqueeze(1)  # (B, T)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, key_padding_mask=mask)

        x = self.dropout(x)

        # CTC head
        logits = F.log_softmax(self.ctc_head(x), dim=-1).permute(1, 0, 2)  # (T, B, vocab)

        # Projection head
        projections = self.proj_head(x, adj_lengths)

        return logits, projections, adj_lengths


# ── Losses (same as Mamba model) ──────────────────────────────────────────────
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.T = temperature

    def forward(self, z, session_ids):
        B = z.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=z.device)
        sim = torch.mm(z, z.T) / self.T
        sim.fill_diagonal_(float('-inf'))
        diff = (session_ids.unsqueeze(0) != session_ids.unsqueeze(1))
        loss = torch.tensor(0.0, device=z.device)
        n = 0
        for i in range(B):
            if diff[i].sum() == 0:
                continue
            loss = loss + (-F.log_softmax(sim[i], dim=0)[diff[i]].mean())
            n += 1
        return loss / max(n, 1)


def blank_penalty(logits, blank_idx=BLANK_IDX, target_ratio=0.7):
    blank_frac = logits.exp()[:, :, blank_idx].mean()
    return torch.clamp(blank_frac - target_ratio, min=0.0)


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, T, N = 4, 300, 124

    model = BCITransformer(n_sessions=N, n_units=512, n_layers=4).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    x    = torch.randn(B, T, 512, device=device)
    sids = torch.randint(0, N, (B,), device=device)
    lens = torch.full((B,), T, device=device)

    model.train()
    with torch.autocast('cuda' if device.type=='cuda' else 'cpu', torch.bfloat16):
        logits, projs, adj = model(x, sids, lens)

    print(f"logits: {logits.shape}")   # (T', B, 41)
    print(f"projs:  {projs.shape}")    # (B, 128)
    print(f"adj_lengths: {adj[0].item()} (from T={T})")

    logits.sum().backward()
    print("Backward OK!")
    print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB" if device.type=='cuda' else "CPU mode")