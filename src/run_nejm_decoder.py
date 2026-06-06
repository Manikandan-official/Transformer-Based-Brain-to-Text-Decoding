import os, json, time, numpy as np, redis
from tqdm import tqdm

YOUR_PHONES = [
    'BLANK','AA','AE','AH','AO','AW','AY','B','CH','D','DH',
    'EH','ER','EY','F','G','HH','IH','IY','JH','K','L','M',
    'N','NG','OW','OY','P','R','S','SH','T','TH','UH','UW',
    'V','W','Y','Z','ZH','SIL'
]
LM_PHONES = [
    '<blk>','SIL','AA','AE','AH','AO','AW','AY','B','CH','D','DH',
    'EH','ER','EY','F','G','HH','IH','IY','JH','K','L','M',
    'N','NG','OW','OY','P','R','S','SH','T','TH','UH','UW',
    'V','W','Y','Z','ZH'
]
REORDER = np.array([YOUR_PHONES.index('BLANK' if p=='<blk>' else p) for p in LM_PHONES])

def reorder_logits(x): return x[:, REORDER]

def compute_wer(hyps, refs):
    te, tw = 0, 0
    for h, r in zip(hyps, refs):
        if not r.strip(): continue
        hw, rw = h.split(), r.split()
        m, n = len(hw), len(rw)
        dp = list(range(n+1))
        for i in range(1, m+1):
            prev, dp[0] = dp[0], i
            for j in range(1, n+1):
                tmp = dp[j]
                dp[j] = prev if hw[i-1]==rw[j-1] else 1+min(prev,dp[j],dp[j-1])
                prev = tmp
        te += dp[n]; tw += len(rw)
    return te/max(tw,1)

def get_ms(r):
    t = r.time()
    return int(t[0]*1000 + t[1]/1000)

def decode_one(r, logits, last_seen, timeout=60):
    r.xadd('remote_lm_reset', {'reset': 1})
    deadline = time.time() + timeout
    while time.time() < deadline:
        ack = r.xread({'remote_lm_done_resetting': last_seen['reset_ack']}, count=1, block=300)
        if ack:
            last_seen['reset_ack'] = ack[0][1][-1][0]
            break
    lm_logits = reorder_logits(logits.astype(np.float32))
    r.xadd('remote_lm_input', {'logits': lm_logits.tobytes()})
    time.sleep(0.02)
    r.xadd('remote_lm_finalize', {'finalize': 1})
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = r.xread({'remote_lm_output_final': last_seen['final_out']}, count=1, block=500)
        if result:
            entry_id, data = result[0][1][0]
            last_seen['final_out'] = entry_id
            decoded = data.get(b'lm_response_final', b'').decode().strip()
            deadline2 = time.time() + 15
            while time.time() < deadline2:
                ack = r.xread({'remote_lm_done_finalizing': last_seen['finalize_ack']}, count=1, block=300)
                if ack:
                    last_seen['finalize_ack'] = ack[0][1][-1][0]
                    break
            return decoded.lower(), last_seen
    return '', last_seen

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--logits_dir', default='logits_t15')
    p.add_argument('--out', default='wer_results.json')
    args = p.parse_args()

    print("Loading logits...")
    logits_arr = np.load(os.path.join(args.logits_dir, 'logits_scaled.npy'), allow_pickle=True)
    with open(os.path.join(args.logits_dir, 'ref_texts.txt')) as f:
        refs = [l.strip() for l in f]
    print(f"Loaded {len(logits_arr)} samples, shape: {logits_arr[0].shape}")
    print(f"blank_frac: {np.exp(logits_arr[0])[:,0].mean():.3f}")

    r = redis.Redis(host='localhost', port=6379)
    r.ping()
    print("Redis connected!")

    for s in ['remote_lm_input','remote_lm_output_final','remote_lm_finalize',
              'remote_lm_reset','remote_lm_done_resetting','remote_lm_done_finalizing']:
        try: r.delete(s)
        except: pass
    time.sleep(0.3)

    print("Waiting for LM server...")
    for i in range(120):
        if r.xlen('remote_lm_args') > 0:
            print("LM server ready!")
            break
        time.sleep(1)

    now = str(get_ms(r))
    last_seen = {'reset_ack': now, 'finalize_ack': now, 'final_out': now}

    hyps, failed = [], 0
    t0 = time.time()
    for i, logits in enumerate(tqdm(logits_arr, desc='Decoding')):
        try:
            decoded, last_seen = decode_one(r, logits.astype(np.float32), last_seen)
            hyps.append(decoded)
            if not decoded: failed += 1
        except Exception as e:
            hyps.append(''); failed += 1

        if (i+1) % 50 == 0:
            pwer = compute_wer(hyps, refs[:i+1])
            print(f"\n[{i+1}] WER={pwer*100:.2f}% hyp='{hyps[-1][:50]}'")

    wer = compute_wer(hyps, refs)
    print(f"\n{'='*50}")
    print(f"FINAL WER: {wer*100:.2f}%")
    print(f"Failed: {failed}/{len(hyps)}")
    print(f"Time: {(time.time()-t0)/60:.1f} min")
    print(f"\nNEJM GRU: 7.44% | Your Mamba: {wer*100:.2f}%")

    for i in range(min(10, len(hyps))):
        if refs[i].strip():
            print(f"Ref: {refs[i]}\nHyp: {hyps[i]}\n")

    with open(args.out,'w') as f:
        json.dump({'WER':wer,'failed':failed,'model_PER':10.64,
                   'hyps':hyps[:100],'refs':refs[:100]},f,indent=2)
    print(f"Saved {args.out}")

if __name__=='__main__':
    main()
