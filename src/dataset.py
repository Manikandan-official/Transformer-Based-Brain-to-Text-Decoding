"""
dataset_multi.py — Multi-subject dataset loader (T12 + T15)
============================================================
Extends dataset.py to handle both T12 and T15 sessions.
Key additions:
  - Subject-aware session IDs (T15: 0-44, T12: 100+)
  - SubjectAdapter in model handles cross-subject normalization
  - Same interface as original dataset.py
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ── Phoneme vocabulary (shared across T12 and T15) ───────────────────────────
PHONEME_LIST = [
    'BLANK', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M',
    'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW',
    'V', 'W', 'Y', 'Z', 'ZH', 'SIL'
]
N_PHONEMES = len(PHONEME_LIST)
BLANK_IDX  = 0


def get_global_session_map(data_root):
    """
    Returns dict: session_name → session_id
    T15 sessions: 0-44
    T12 sessions: 100+ (to avoid conflict)
    """
    hdf5_root = os.path.join(data_root, 'hdf5_data_final')
    t15 = sorted([d for d in os.listdir(hdf5_root) if d.startswith('t15.')])
    t12 = sorted([d for d in os.listdir(hdf5_root) if d.startswith('t12.')])

    session_map = {}
    for i, s in enumerate(t15):
        session_map[s] = i
    for i, s in enumerate(t12):
        session_map[s] = 100 + i

    return session_map


def discover_sessions(data_root, split='train'):
    """Returns list of (hdf5_path, session_id, session_name)."""
    hdf5_root   = os.path.join(data_root, 'hdf5_data_final')
    session_map = get_global_session_map(data_root)
    sessions    = []

    for session in sorted(os.listdir(hdf5_root)):
        if not (session.startswith('t15.') or session.startswith('t12.')):
            continue
        path = os.path.join(hdf5_root, session, f'data_{split}.hdf5')
        if not os.path.exists(path):
            continue
        sid = session_map[session]
        sessions.append((path, sid, session))

    return sessions


class MultiSubjectBCIDataset(Dataset):
    """
    Combined T12 + T15 dataset.
    Returns: (neural, n_len, phones, p_len, sid, subject_id)
    subject_id: 0=T15, 1=T12
    """
    def __init__(self, data_root, split='train', max_sessions=None):
        self.trials = []
        sessions = discover_sessions(data_root, split)

        if max_sessions:
            sessions = sessions[:max_sessions]

        print(f"[Dataset] '{split}' — {len(sessions)} session files found")
        for path, sid, name in sessions:
            subject = 0 if name.startswith('t15.') else 1
            try:
                with h5py.File(path, 'r') as f:
                    keys = sorted(f.keys())
                    for key in keys:
                        if not key.startswith('trial_'):
                            continue
                        self.trials.append({
                            'path':    path,
                            'key':     key,
                            'sid':     sid,
                            'subject': subject,
                        })
                print(f"  {name:30s} → {len(keys):4d} trials "
                      f"(session_id={sid}, subject={'T15' if subject==0 else 'T12'})")
            except Exception as e:
                print(f"  {name}: ERROR {e}")

        print(f"[Dataset] Total: {len(self.trials)} trials "
              f"(T15+T12 combined)")

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        t = self.trials[idx]
        with h5py.File(t['path'], 'r') as f:
            grp      = f[t['key']]
            neural   = torch.tensor(grp['input_features'][:],
                                    dtype=torch.float32)
            phones   = torch.tensor(grp['seq_class_ids'][:],
                                    dtype=torch.long)

        # Get actual phoneme length (non-zero)
        p_len  = int((phones > 0).sum())
        phones = torch.clamp(phones, 1, N_PHONEMES - 1)
        n_len  = neural.shape[0]

        return (neural, n_len, phones, p_len,
                t['sid'], t['subject'])


def collate_fn(batch):
    """Pad variable-length sequences."""
    neurals, n_lens, phones_list, p_lens, sids, subjects = zip(*batch)

    max_t = max(n.shape[0] for n in neurals)
    max_p = max(p.shape[0] for p in phones_list)

    B = len(neurals)
    neural_pad = torch.zeros(B, max_t, neurals[0].shape[1])
    phone_pad  = torch.zeros(B, max_p, dtype=torch.long)

    for i, (n, p) in enumerate(zip(neurals, phones_list)):
        neural_pad[i, :n.shape[0]] = n
        phone_pad[i,  :p.shape[0]] = p

    return (
        neural_pad,
        torch.tensor(n_lens, dtype=torch.long),
        phone_pad,
        torch.tensor(p_lens, dtype=torch.long),
        torch.tensor(sids,     dtype=torch.long),
        torch.tensor(subjects, dtype=torch.long),
    )


class BalancedSubjectSampler(torch.utils.data.Sampler):
    """Batch sampler: each batch = 50% T15 + 50% T12."""
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.half = batch_size // 2
        self.indices_t15 = [i for i, t in enumerate(dataset.trials) if t["subject"] == 0]
        self.indices_t12 = [i for i, t in enumerate(dataset.trials) if t["subject"] == 1]
        self.n_batches = max(len(self.indices_t15), len(self.indices_t12)) // self.half

    def __iter__(self):
        import random
        t15 = self.indices_t15.copy()
        t12 = self.indices_t12.copy()
        random.shuffle(t15)
        random.shuffle(t12)
        while len(t15) < self.n_batches * self.half:
            extra = self.indices_t15.copy()
            random.shuffle(extra)
            t15 += extra
        while len(t12) < self.n_batches * self.half:
            extra = self.indices_t12.copy()
            random.shuffle(extra)
            t12 += extra
        for i in range(self.n_batches):
            batch = t15[i*self.half:(i+1)*self.half] + t12[i*self.half:(i+1)*self.half]
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


def get_dataloaders(data_root, batch_size=8, num_workers=0,
                    max_sessions=None):
    train_ds = MultiSubjectBCIDataset(data_root, 'train', max_sessions)
    val_ds   = MultiSubjectBCIDataset(data_root, 'val',   max_sessions)

    # Count sessions for model
    session_map = get_global_session_map(data_root)
    n_sessions  = max(session_map.values()) + 1  # covers both T12 and T15

    sampler = BalancedSubjectSampler(train_ds, batch_size)
    train_dl = DataLoader(train_ds, batch_sampler=sampler,
                          num_workers=num_workers,
                          collate_fn=collate_fn, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size,
                          shuffle=False, num_workers=num_workers,
                          collate_fn=collate_fn, pin_memory=True)

    return train_dl, val_dl, n_sessions

