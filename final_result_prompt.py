import json

with open('eval.json') as f:
    rows = [json.loads(line) for line in f]

# find a row with at least one perfectly correct sample
for row in rows:
    for i, score in enumerate(row['scores']):
        if score == 1.0:
            print(f"=== Prompt ===\n{row['prompt']}")
            print(f"\n=== Target: {row['target']}, Numbers: {row['nums']} ===")
            print(f"\n=== Model response ===\n{row['response'][i]}")
            print(f"\n=== Score: {score} ===")
            break
    else:
        continue
    break