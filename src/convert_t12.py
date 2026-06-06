"""
Convert T12 .mat files to T15-compatible HDF5 format.
T12: tx1-tx4 (256 each) + spikePow (256) → use tx1+tx2 (512) to match T15
"""
import os
import numpy as np
import scipy.io as sio
import h5py
from g2p_en import G2p

# NEJM phoneme list — must match dataset.py exactly
PHONEME_LIST = [
    'BLANK', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M',
    'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW',
    'V', 'W', 'Y', 'Z', 'ZH', 'SIL'
]
PHONE_TO_IDX = {p: i for i, p in enumerate(PHONEME_LIST)}

g2p = G2p()

def text_to_phonemes(text):
    """Convert text to phoneme indices using g2p."""
    text = text.strip().lower()
    phones = g2p(text)
    indices = []
    for p in phones:
        if p == ' ':
            indices.append(PHONE_TO_IDX['SIL'])
        else:
            # Strip stress markers (0,1,2)
            p_clean = p.rstrip('012').upper()
            if p_clean in PHONE_TO_IDX:
                indices.append(PHONE_TO_IDX[p_clean])
    return indices

def convert_mat_to_hdf5(mat_path, out_train, out_val, val_ratio=0.1):
    """Convert one .mat session to train/val HDF5 files."""
    d = sio.loadmat(mat_path)
    
    tx1 = d['tx1'][0]  # array of trials
    tx2 = d['tx2'][0]
    texts = d['sentenceText']  # (N,) array of strings
    n_trials = len(tx1)
    
    # Split train/val
    n_val = max(1, int(n_trials * val_ratio))
    n_train = n_trials - n_val
    
    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    
    def write_trials(hdf5_path, trial_indices):
        with h5py.File(hdf5_path, 'w') as f:
            for out_idx, trial_idx in enumerate(trial_indices):
                # Neural features: tx1 + tx2 = 512 features
                feat1 = tx1[trial_idx].astype(np.float32)  # (T, 256)
                feat2 = tx2[trial_idx].astype(np.float32)  # (T, 256)
                neural = np.concatenate([feat1, feat2], axis=1)  # (T, 512)
                
                # Text
                text = str(texts[trial_idx]).strip()
                
                # Phonemes
                phone_ids = text_to_phonemes(text)
                seq_class = np.zeros(500, dtype=np.int32)
                seq_class[:len(phone_ids)] = phone_ids[:500]
                
                # Transcription as ASCII
                trans = np.zeros(500, dtype=np.int32)
                for ci, c in enumerate(text[:500]):
                    trans[ci] = ord(c)
                
                key = f'trial_{out_idx:04d}'
                grp = f.create_group(key)
                grp.create_dataset('input_features', data=neural)
                grp.create_dataset('seq_class_ids', data=seq_class)
                grp.create_dataset('transcription', data=trans)
    
    write_trials(out_train, range(n_train))
    write_trials(out_val, range(n_train, n_trials))
    
    return n_train, n_val

def main():
    mat_dir = 'competitionData/train'
    out_root = 'hdf5_data_final'
    
    mat_files = sorted([f for f in os.listdir(mat_dir) if f.endswith('.mat')])
    print(f"Found {len(mat_files)} T12 sessions")
    
    total_train, total_val = 0, 0
    for mat_file in mat_files:
        session = mat_file.replace('.mat', '')  # e.g. t12.2022.04.28
        mat_path = os.path.join(mat_dir, mat_file)
        out_dir = os.path.join(out_root, session)
        os.makedirs(out_dir, exist_ok=True)
        
        out_train = os.path.join(out_dir, 'data_train.hdf5')
        out_val = os.path.join(out_dir, 'data_val.hdf5')
        
        n_train, n_val = convert_mat_to_hdf5(mat_path, out_train, out_val)
        print(f"  {session}: {n_train} train, {n_val} val")
        total_train += n_train
        total_val += n_val
    
    print(f"\nTotal T12: {total_train} train, {total_val} val")
    print("Done! T12 sessions added to hdf5_data_final/")

if __name__ == '__main__':
    main()
