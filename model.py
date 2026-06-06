import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
from dataset import N_PHONEMES, BLANK_IDX

def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class SoftWindowBiMamba(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, dt_min=0.025, dt_max=1.0):
        super().__init__()
        self.fwd = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dt_min=dt_min, dt_max=dt_max)
        self.bwd = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dt_min=dt_min, dt_max=dt_max)
        self._force_short_memory(self.fwd)
        self._force_short_memory(self.bwd)

    def _force_short_memory(self, m):
        if hasattr(m, 'dt_bias'):
            with torch.no_grad(): m.dt_bias.add_(1.0)
        elif hasattr(m, 'dt_proj'):
            with torch.no_grad(): m.dt_proj.bias.add_(1.0)

    def forward(self, x):
        return self.fwd(x) + self.bwd(x.flip(1)).flip(1)

class DayAdapter(nn.Module):
    def __init__(self, n_sessions, input_dim=512):
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.eye(input_dim)) for _ in range(n_sessions)])
        self.biases  = nn.ParameterList([nn.Parameter(torch.zeros(1, input_dim)) for _ in range(n_sessions)])
        self.act = nn.Softsign()

    def forward(self, x, session_ids):
        out = torch.zeros_like(x)
        for sid in session_ids.unique():
            mask = (session_ids == sid)
            w = self.weights[sid.item()]
            b = self.biases[sid.item()]
            out[mask] = self.act(torch.einsum('btd,dk->btk', x[mask], w) + b).to(x.dtype)
        return out

class PatchEmbedding(nn.Module):
    def __init__(self, input_dim, d_model, patch_size=16, stride=4):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.proj = nn.Sequential(
            nn.Linear(input_dim * patch_size, 4096),
            nn.Softsign(),
            nn.Dropout(0.3),
            nn.Linear(4096, d_model),
        )

    def forward(self, x):
        B, T, C = x.shape
        x = x.unsqueeze(1).permute(0, 3, 1, 2)
        x = x.unfold(3, self.patch_size, self.stride)
        x = x.squeeze(2).permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        return self.proj(x)

    def output_lengths(self, input_lengths):
        return ((input_lengths.float() - self.patch_size) / self.stride + 1).long().clamp(min=1)

class ProjectionHead(nn.Module):
    def __init__(self, d_model, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, proj_dim))

    def forward(self, x, lengths):
        mask = (torch.arange(x.shape[1], device=x.device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1).float()
        pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        return F.normalize(self.net(pooled), dim=-1)

class BCIDecoder(nn.Module):
    def __init__(self, n_sessions, input_dim=512, n_units=512, n_layers=4,
                 d_state=64, patch_size=16, patch_stride=4,
                 dropout=0.4, drop_path_rate=0.2, proj_dim=128, use_checkpoint=False):
        super().__init__()
        self.day_adapter  = DayAdapter(n_sessions, input_dim)
        self.patch_embed  = PatchEmbedding(input_dim, n_units, patch_size, patch_stride)
        self.layers       = nn.ModuleList([SoftWindowBiMamba(n_units, d_state=d_state) for _ in range(n_layers)])
        self.norms        = nn.ModuleList([nn.LayerNorm(n_units) for _ in range(n_layers)])
        self.drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.dropout      = nn.Dropout(dropout)
        self.ctc_head     = nn.Linear(n_units, N_PHONEMES)
        self.proj_head    = ProjectionHead(n_units, proj_dim)
        nn.init.xavier_uniform_(self.ctc_head.weight)

    def forward(self, x, session_ids, lengths):
        x = self.day_adapter(x, session_ids)
        x = self.patch_embed(x)
        adj_lengths = self.patch_embed.output_lengths(lengths)
        for i, (norm, layer) in enumerate(zip(self.norms, self.layers)):
            x_norm = norm(x)
            layer_out = layer(x_norm)
            layer_out = drop_path(layer_out, self.drop_path_rates[i], self.training)
            x = x + layer_out
        x = self.dropout(x)
        logits = F.log_softmax(self.ctc_head(x), dim=-1).permute(1, 0, 2)
        projections = self.proj_head(x, adj_lengths)
        return logits, projections, adj_lengths

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
            if diff[i].sum() == 0: continue
            loss = loss + (-F.log_softmax(sim[i], dim=0)[diff[i]].mean())
            n += 1
        return loss / max(n, 1)

def blank_penalty(logits, blank_idx=BLANK_IDX, target_ratio=0.7):
    blank_frac = logits.exp()[:, :, blank_idx].mean()
    return torch.clamp(blank_frac - target_ratio, min=0.0)
