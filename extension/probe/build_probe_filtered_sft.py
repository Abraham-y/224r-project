"""Build SFT data filtered by probe (Option 1: probe-as-SFT-data-filter).

Steps:
  1. Load the C_outcome-trained L16 pre_answer probe (Pipeline pickle).
  2. Load C_SFT pre_answer hidden states (cached) + corresponding rollout texts.
  3. For each prompt's K rollouts, compute probe scores; take rollouts that are
     verifier-correct AND in the top-half by probe score.
  4. Save as JSONL (prompt + response) for SFT.
"""
import json, os, pickle, re, sys
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from evaluation.countdown import validate_equation, evaluate_equation


def score(eq, gt):
    if not validate_equation(eq, gt['numbers']): return 0.1
    try:
        r = evaluate_equation(eq)
        return 1.0 if r is not None and abs(r - gt['target']) < 1e-5 else 0.1
    except: return 0.1


# 1. Load probe (Pipeline = StandardScaler -> LogisticRegression)
PROBE_PATH = 'extension/cache/steering/probe_pipeline_C_outcome_l16_pre_answer.pkl'
with open(PROBE_PATH, 'rb') as f:
    probe = pickle.load(f)
print(f"loaded probe: {type(probe).__name__}")


# 2. Load C_SFT hidden states + meta (this gives us prompt_idx, rollout_idx per row)
HS_PATH = 'extension/cache/probe_cache_n500_clean406/C_SFT_l16_pre_answer.npz'
META_PATH = 'extension/cache/probe_cache_n500_clean406/C_SFT_l16_pre_answer.meta.json'
hs = np.load(HS_PATH)['X']
with open(META_PATH) as f:
    meta = json.load(f)
print(f"hidden states: {hs.shape}, n_meta rows: {len(meta)}")


# 3. Load eval JSON for the rollout text
EVAL_PATH = 'eval_c_sft_n500.json'
eval_rows = []
with open(EVAL_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            try: eval_rows.append(json.loads(line))
            except: pass
print(f"eval rows: {len(eval_rows)}")


# 4. Probe scoring for each cached hidden state
probe_scores = probe.predict_proba(hs)[:, 1]  # P(correct)
print(f"probe scores: mean={probe_scores.mean():.3f} std={probe_scores.std():.3f}")


# 5. Build (prompt, response) SFT data
# Each meta row has prompt_idx, rollout_idx pointing into eval_rows
out_rows = []
n_correct = 0
n_topprobe = 0
n_kept = 0
# Group meta+probe by prompt_idx
from collections import defaultdict
by_prompt = defaultdict(list)
for i, m in enumerate(meta):
    by_prompt[m['prompt_idx']].append((m['resp_idx'], probe_scores[i]))

for pidx, rollout_list in by_prompt.items():
    if pidx >= len(eval_rows):
        continue
    er = eval_rows[pidx]
    prompt = er['prompt']
    responses = er.get('response', [])
    gt = er.get('ground_truth') or {'target': er['target'], 'numbers': er['nums']}
    # For each rollout, compute verifier score + probe score
    annotated = []
    for r_idx, pscore in rollout_list:
        if r_idx >= len(responses): continue
        resp = responses[r_idx]
        m = re.search(r'<answer>(.*?)</answer>', resp, re.DOTALL)
        if not m:
            continue
        first_eq = m.group(1).strip()
        v = score(first_eq, gt)
        annotated.append((r_idx, pscore, v, resp))
    if not annotated: continue
    # Filter: verifier-correct
    correct = [x for x in annotated if x[2] == 1.0]
    n_correct += len(correct)
    if not correct: continue
    # Among correct, take top half by probe
    correct.sort(key=lambda x: -x[1])
    keep = correct[:max(1, len(correct) // 2)]
    n_kept += len(keep)
    for (_, pscore, _, resp) in keep:
        out_rows.append({
            'prompt': prompt,
            'response': resp,
            'probe_score': float(pscore),
            'prompt_idx': int(pidx),
        })

print(f"verifier-correct rollouts: {n_correct}")
print(f"probe-top-half of correct: {n_kept}")
print(f"final SFT examples: {len(out_rows)}")

OUT = 'extension/data/sft_probe_filtered_csft.jsonl'
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    for r in out_rows:
        f.write(json.dumps(r) + '\n')
print(f"wrote {OUT}")
