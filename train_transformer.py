"""
train_transformer.py — Train Transformer BCI Decoder
=====================================================
Same training loop as train.py but uses BCITransformer instead of BCIDecoder.
After training, ensemble with Mamba logits for lower WER.

Usage:
    python train_transformer.py \
        --data_root data --epochs 80 --batch_size 8 \
        --model_dim 512 --n_layers 4 --n_heads 8 \
        --lr 1e-4 --lr_day 3e-4 \
        --ckpt_dir checkpoints_transformer
"""

import os, argparse, math, time, copy
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from dataset import get_dataloaders, BLANK_IDX, N_PHONEMES
from model_transformer import BCITransformer, NTXentLoss
from evaluate import compute_per, greedy_decode, edit_distance


def gauss_smooth(x, kernel_std=2.0, kernel_size=11, device='cpu'):
    half   = kernel_size // 2
    coords = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-0.5 * (coords / kernel_std) ** 2)
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    B, T, C = x.shape
    x_t     = x.permute(0, 2, 1).reshape(B * C, 1, T)
    return F.conv1d(x_t, kernel, padding=half).view(B, C, T).permute(0, 2, 1)


def augment(x, cfg, device, epoch=1):
    B, T, C = x.shape
    if cfg.get('static_gain_std', 0) > 0:
        warp  = torch.eye(C, device=device).unsqueeze(0).expand(B, -1, -1).clone()
        warp += torch.randn(B, C, C, device=device) * cfg['static_gain_std']
        x     = torch.bmm(x, warp)
    if cfg.get('white_noise_std', 0) > 0:
        x = x + torch.randn_like(x) * cfg['white_noise_std']
    if cfg.get('const_offset_std', 0) > 0:
        x = x + torch.randn(B, 1, C, device=device) * cfg['const_offset_std']
    if cfg.get('random_walk_std', 0) > 0:
        x = x + torch.cumsum(
            torch.randn(B, T, C, device=device) * cfg['random_walk_std'], dim=1)
    if cfg.get('time_mask_ratio', 0) > 0 and epoch > 2:
        max_mask = int(T * cfg['time_mask_ratio'])
        for b in range(B):
            mask_len   = torch.randint(0, max_mask + 1, (1,)).item()
            mask_start = torch.randint(0, max(1, T - mask_len), (1,)).item()
            x[b, mask_start:mask_start + mask_len] = 0.0
    return x


def ctc_loss_smoothed(ctc_fn, logits, targets_flat, input_lens, target_lens, smooth_eps=0.1):
    l_hard = ctc_fn(logits, targets_flat, input_lens, target_lens)
    if smooth_eps <= 0:
        return l_hard
    return (1 - smooth_eps) * l_hard + smooth_eps * (-logits.mean())


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
        for ema_b, b in zip(self.model.buffers(), model.buffers()):
            ema_b.copy_(b)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',       default='data')
    p.add_argument('--ckpt_dir',        default='./checkpoints_transformer')
    p.add_argument('--epochs',          type=int,   default=80)
    p.add_argument('--batch_size',      type=int,   default=8)
    p.add_argument('--model_dim',       type=int,   default=512)
    p.add_argument('--n_layers',        type=int,   default=4)
    p.add_argument('--n_heads',         type=int,   default=8)
    p.add_argument('--dim_feedforward', type=int,   default=2048)
    p.add_argument('--lr',              type=float, default=1e-4)
    p.add_argument('--lr_day',          type=float, default=3e-4)
    p.add_argument('--lr_min',          type=float, default=1e-6)
    p.add_argument('--warmup_frac',     type=float, default=0.02)
    p.add_argument('--decay_frac',      type=float, default=0.8)
    p.add_argument('--grad_clip',       type=float, default=1.0)
    p.add_argument('--lam',             type=float, default=0.01)
    p.add_argument('--dropout',         type=float, default=0.35)
    p.add_argument('--smooth_eps',      type=float, default=0.1)
    p.add_argument('--ema_decay',       type=float, default=0.999)
    p.add_argument('--use_amp',         type=lambda x: str(x).lower() != 'false', default=True)
    p.add_argument('--aug_static_gain',  type=float, default=0.05)
    p.add_argument('--aug_white_noise',  type=float, default=0.01)
    p.add_argument('--aug_const_offset', type=float, default=0.01)
    p.add_argument('--aug_random_walk',  type=float, default=0.002)
    p.add_argument('--aug_time_mask',    type=float, default=0.10)
    p.add_argument('--smooth_std',      type=float, default=2.0)
    p.add_argument('--smooth_size',     type=int,   default=11)
    p.add_argument('--patience',        type=int,   default=20)
    p.add_argument('--num_workers',     type=int,   default=0)
    p.add_argument('--max_sessions',    type=int,   default=None)
    return p.parse_args()


def make_lr_lambda(lr_min_ratio, total_steps, warmup_steps, decay_steps):
    def fn(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if step < decay_steps:
            progress = (step - warmup_steps) / max(decay_steps - warmup_steps, 1)
            cosine   = 0.5 * (1 + math.cos(math.pi * progress))
            return max(lr_min_ratio, lr_min_ratio + (1 - lr_min_ratio) * cosine)
        return lr_min_ratio
    return fn


def train_epoch(model, loader, opt, sched, ctc_fn, contrastive_loss,
                lam, aug_cfg, smooth_cfg, device, grad_clip, use_amp,
                smooth_eps, epoch):
    model.train()
    totals = {'loss': 0, 'ctc': 0, 'cont': 0}
    n = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == 'cuda')

    for neural, n_lens, phones, p_lens, sids, _ in tqdm(loader, desc='train', leave=False):
        neural = neural.to(device, non_blocking=True)
        sids   = sids.to(device)
        n_lens = n_lens.to(device)
        p_lens = p_lens.to(device)
        phones_flat = torch.cat([phones[i, :p_lens[i]] for i in range(len(p_lens))]).to(device)

        opt.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=use_amp and device.type == 'cuda'):
            neural = augment(neural, aug_cfg, device, epoch=epoch)
            neural = gauss_smooth(neural, smooth_cfg['std'], smooth_cfg['size'], device)
            logits, projs, adj_lens = model(neural, sids, n_lens)
            l_ctc  = ctc_loss_smoothed(ctc_fn, logits, phones_flat, adj_lens, p_lens, smooth_eps).mean()
            l_cont = contrastive_loss(projs, sids)
            loss   = l_ctc + lam * l_cont

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()

        totals['loss'] += loss.item()
        totals['ctc']  += l_ctc.item()
        totals['cont'] += l_cont.item()
        n += 1

    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def val_epoch(model, loader, ctc_fn, contrastive_loss, lam,
              smooth_cfg, device, use_amp, n_sessions):
    model.eval()
    total_loss, n = 0, 0
    session_stats = {sid: {'edit': 0, 'total': 0} for sid in range(n_sessions)}

    for neural, n_lens, phones, p_lens, sids, _ in tqdm(loader, desc='val  ', leave=False):
        neural = neural.to(device, non_blocking=True)
        sids   = sids.to(device)
        n_lens = n_lens.to(device)
        p_lens = p_lens.to(device)
        phones_flat = torch.cat([phones[i, :p_lens[i]] for i in range(len(p_lens))]).to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=use_amp and device.type == 'cuda'):
            neural   = gauss_smooth(neural, smooth_cfg['std'], smooth_cfg['size'], device)
            logits, projs, adj_lens = model(neural, sids, n_lens)
            l_ctc  = ctc_fn(logits, phones_flat, adj_lens, p_lens).mean()
            l_cont = contrastive_loss(projs, sids)
            loss   = l_ctc + lam * l_cont

        total_loss += loss.item()
        n += 1

        decoded = greedy_decode(logits.float().cpu(), adj_lens.cpu())
        for i, sid in enumerate(sids.cpu().tolist()):
            ref = phones[i, :p_lens[i]].tolist()
            if ref:
                session_stats[sid]['edit']  += edit_distance(decoded[i], ref)
                session_stats[sid]['total'] += len(ref)

    val_PER = sum(s['edit'] for s in session_stats.values()) / max(
              sum(s['total'] for s in session_stats.values()), 1)
    session_per = {
        sid: (s['edit'] / s['total']) if s['total'] > 0 else float('nan')
        for sid, s in session_stats.items()
    }
    return {'val_loss': total_loss / max(n, 1), 'val_PER': val_PER, 'session_PER': session_per}


def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  AMP: {args.use_amp}")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    aug_cfg    = dict(static_gain_std=args.aug_static_gain, white_noise_std=args.aug_white_noise,
                      const_offset_std=args.aug_const_offset, random_walk_std=args.aug_random_walk,
                      time_mask_ratio=args.aug_time_mask)
    smooth_cfg = dict(std=args.smooth_std, size=args.smooth_size)

    tl, vl, n_sess = get_dataloaders(args.data_root, args.batch_size, args.num_workers, args.max_sessions)
    print(f"Sessions: {n_sess}, Train batches: {len(tl)}")

    total_steps  = len(tl) * args.epochs
    warmup_steps = max(100, int(total_steps * args.warmup_frac))
    decay_steps  = int(total_steps * args.decay_frac)

    model = BCITransformer(
        n_sessions=n_sess, n_units=args.model_dim, n_layers=args.n_layers,
        n_heads=args.n_heads, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    day_params   = [p for nm, p in model.named_parameters() if 'day_adapter' in nm]
    other_params = [p for nm, p in model.named_parameters() if 'day_adapter' not in nm]
    opt = optim.AdamW([{'params': day_params, 'lr': args.lr_day},
                       {'params': other_params, 'lr': args.lr}], weight_decay=1e-4)

    sched = LambdaLR(opt, lr_lambda=[
        make_lr_lambda(args.lr_min / args.lr_day, total_steps, warmup_steps, decay_steps),
        make_lr_lambda(args.lr_min / args.lr,     total_steps, warmup_steps, decay_steps),
    ])

    ctc_fn           = torch.nn.CTCLoss(blank=BLANK_IDX, reduction='none', zero_infinity=True).to(device)
    contrastive_loss = NTXentLoss().to(device)

    best_per, patience, history = float('inf'), 0, []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_epoch(model, tl, opt, sched, ctc_fn, contrastive_loss,
                              args.lam, aug_cfg, smooth_cfg, device, args.grad_clip,
                              args.use_amp, smooth_eps=args.smooth_eps, epoch=epoch)
        val_m = val_epoch(model, vl, ctc_fn, contrastive_loss, args.lam,
                          smooth_cfg, device, args.use_amp, n_sess)
        if ema: ema.update(model)

        # LIVE blank_frac
        model.eval()
        with torch.no_grad():
            for neural, n_lens, phones, p_lens, sids, _ in vl:
                neural = gauss_smooth(neural.to(device), smooth_cfg['std'], smooth_cfg['size'], device)
                logits, _, adj = model(neural.to(device), sids.to(device), n_lens.to(device))
                bf = logits.exp()[:,:,BLANK_IDX].mean().item()
                print(f"  LIVE blank_frac={bf:.3f}")
                break
        model.train()

        dur = time.time() - t0
        print(f"\nEpoch {epoch:3d} ({dur:.0f}s) | "
              f"loss={train_m['loss']:.4f} ctc={train_m['ctc']:.4f} | "
              f"val_PER={val_m['val_PER']:.4f}")

        # T15 vs T12 PER
        t15_stats = {sid: s for sid, s in
                     {sid: {'edit': 0, 'total': 0} for sid in range(n_sess)}.items()
                     if sid < 45}

        if val_m['val_PER'] < best_per:
            best_per, patience = val_m['val_PER'], 0
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'ema_model': ema.model.state_dict() if ema else None,
                'val_PER': best_per, 'session_PER': val_m['session_PER'],
                'args': {**vars(args), 'n_sessions': n_sess},
            }, os.path.join(args.ckpt_dir, 'best.pt'))
            print(f"  ✓ New best PER={best_per:.4f}")
        else:
            patience += 1
            print(f"  No improvement ({patience}/{args.patience})")

        if epoch % 5 == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict()},
                       os.path.join(args.ckpt_dir, f'epoch_{epoch:03d}.pt'))

        if patience >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nDone. Best val PER: {best_per:.4f}")


if __name__ == '__main__':
    main()