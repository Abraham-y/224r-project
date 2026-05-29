import json

with open('eval.json') as f:
    rows = [json.loads(line) for line in f]

print(f"Number of prompts: {len(rows)}")
print(f"Keys in each row: {list(rows[0].keys())}")
print(f"Samples per prompt: {len(rows[0]['scores'])}")

all_scores = [s for row in rows for s in row['scores']]
avg_score = sum(all_scores) / len(all_scores)

# pass@k: fraction of prompts where at least one of K samples scores 1.0
k = len(rows[0]['scores'])
pass_at_k = sum(1 for row in rows if any(s == 1.0 for s in row['scores'])) / len(rows)

# pass@1: average of any single sample being correct
pass_at_1_correct = sum(1 for row in rows for s in row['scores'] if s == 1.0) / len(all_scores)

print(f"\n=== Results ===")
print(f"Average score (raw, includes 0.1 partial): {avg_score:.3f}")
print(f"Pass@1 (strict, score == 1.0): {pass_at_1_correct:.3f}")
print(f"Pass@{k}: {pass_at_k:.3f}")
print(f"\nMilestone score (avg / 0.3 capped at 100%): {min(avg_score / 0.3, 1.0) * 100:.1f}%")
print(f"With 5% margin (0.25 threshold): {min(avg_score / 0.25, 1.0) * 100:.1f}%")