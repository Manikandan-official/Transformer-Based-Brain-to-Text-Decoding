"""evaluate.py — Greedy CTC decode + Phoneme Error Rate"""

import torch
import numpy as np
from dataset import BLANK_IDX, N_PHONEMES, PHONEME_LIST


def greedy_decode(logits, lengths):
    """
    logits  : (T, B, N_PHONEMES)
    lengths : (B,)
    Returns list of decoded phoneme-index lists (one per sample).
    """
    pred = logits.argmax(-1).permute(1, 0)   # (B, T)
    out  = []
    for i, L in enumerate(lengths.tolist()):
        seq, prev = [], None
        for tok in pred[i, :int(L)].tolist():
            if tok != prev:
                if tok != BLANK_IDX:
                    seq.append(tok)
                prev = tok
        out.append(seq)
    return out


def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp   = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev  = tmp
    return dp[n]


def compute_per(logits, targets, input_lengths, target_lengths):
    """Returns list of per-sample PER floats."""
    decoded = greedy_decode(logits, input_lengths)
    pers    = []
    for i, tlen in enumerate(target_lengths.tolist()):
        ref = targets[i, :int(tlen)].tolist()
        if not ref:
            continue
        pers.append(edit_distance(decoded[i], ref) / len(ref))
    return pers
