"""SFT on probe-filtered C_SFT rollouts (Option 1: probe-as-SFT-data-filter).

Loads a local JSONL of (prompt, response) pairs that were pre-selected by
extension/probe/build_probe_filtered_sft.py (verifier-correct AND probe-top-half).

Strategy: monkey-patch sft_trainer.sft_dataset.get_dataloaders to construct
dataloaders from the local JSONL instead of a HuggingFace dataset. The prompts
in the JSONL are already chat-templated (they came from eval rollouts), so we
SKIP the chat-template map step entirely -- we treat the prompt as the literal
input prefix and the response as the literal completion.

CLI passes through to sft.py unchanged. New flag:
  --sft_data_jsonl PATH   local JSONL with prompt+response per line
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SFT_DIR = os.path.join(_REPO_ROOT, "sft_trainer")
for _p in (_SFT_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _pop_arg(name: str, default: str | None = None) -> str | None:
    out = []
    found = default
    skip = False
    for i, tok in enumerate(sys.argv):
        if skip:
            skip = False
            continue
        if tok == f"--{name}" and i + 1 < len(sys.argv):
            found = sys.argv[i + 1]
            skip = True
            continue
        out.append(tok)
    sys.argv[:] = out
    return found


def _install_local_jsonl_loader(jsonl_path: str, test_frac: float = 0.05):
    """Patch sft_trainer.sft_dataset.get_dataloaders to load from a local JSONL."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    import sft_trainer.sft_dataset as sd

    print(f"[sft_probe_filtered] loading local JSONL: {jsonl_path}", flush=True)
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    print(f"[sft_probe_filtered] loaded {len(rows)} examples", flush=True)

    # Split by prompt_idx to avoid duplicates leaking train<->test
    by_prompt = {}
    for r in rows:
        pi = r.get("prompt_idx", -1)
        by_prompt.setdefault(pi, []).append(r)
    prompt_ids = sorted(by_prompt.keys())
    import random
    rng = random.Random(0)
    rng.shuffle(prompt_ids)
    n_test = max(1, int(len(prompt_ids) * test_frac))
    test_ids = set(prompt_ids[:n_test])
    train_ids = set(prompt_ids[n_test:])
    train_rows = [r for r in rows if r.get("prompt_idx", -1) in train_ids]
    test_rows = [r for r in rows if r.get("prompt_idx", -1) in test_ids]
    print(f"[sft_probe_filtered] split: train={len(train_rows)} ({len(train_ids)} prompts), "
          f"test={len(test_rows)} ({len(test_ids)} prompts)", flush=True)

    class LocalJSONLDataset(Dataset):
        def __init__(self, rows, tokenizer, max_prompt_length, max_response_length):
            self.rows = rows
            self.tokenizer = tokenizer
            self.max_prompt_length = max_prompt_length
            self.max_response_length = max_response_length

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            r = self.rows[idx]
            return {"prompt": r["prompt"], "response": r["response"]}

        def collate_fn(self, batch):
            prompts = [item["prompt"] for item in batch]
            responses = [item["response"] for item in batch]
            prompt_toks = self.tokenizer(
                prompts, add_special_tokens=False, padding=True, truncation=True,
                max_length=self.max_prompt_length, padding_side="left", return_tensors="pt",
            )
            response_toks = self.tokenizer(
                responses, add_special_tokens=False, padding=True, truncation=True,
                max_length=self.max_response_length, padding_side="right", return_tensors="pt",
            )
            p_ids, p_mask = prompt_toks["input_ids"], prompt_toks["attention_mask"]
            r_ids, r_mask = response_toks["input_ids"], response_toks["attention_mask"]
            # collapse batch dim if single-sample squeeze occurred
            if p_ids.dim() == 1: p_ids = p_ids.unsqueeze(0); p_mask = p_mask.unsqueeze(0)
            if r_ids.dim() == 1: r_ids = r_ids.unsqueeze(0); r_mask = r_mask.unsqueeze(0)
            input_ids = torch.cat([p_ids, r_ids], dim=1)
            attention_mask = torch.cat([p_mask, r_mask], dim=1)
            is_response_token = torch.cat(
                [torch.zeros_like(p_ids), torch.ones_like(r_ids)], dim=1
            )
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "is_response_token": is_response_token,
            }

    def patched_get_dataloaders(
        dataset_name, tokenizer,
        max_prompt_length=512, max_response_length=1024,
        padding=True, truncation=True, num_proc=None, batch_size=16,
        splits=("train", "test"), num_workers=4, pin_memory=True, drop_last=True,
        prompt_key="query", response_key="completion", shuffle=True,
        gradient_accumulation_steps=1,
    ):
        bs = batch_size // gradient_accumulation_steps
        out = {}
        for split in splits:
            rows_for_split = train_rows if split == "train" else test_rows
            ds = LocalJSONLDataset(rows_for_split, tokenizer, max_prompt_length, max_response_length)
            out[split] = DataLoader(
                ds, batch_size=bs, shuffle=(shuffle and split == "train"),
                collate_fn=ds.collate_fn, num_workers=num_workers,
                pin_memory=pin_memory, drop_last=(drop_last and split == "train"),
            )
        return out

    sd.get_dataloaders = patched_get_dataloaders
    # also patch the import in sft.py's namespace if already imported
    try:
        import sft_trainer.sft as _sft
        _sft.get_dataloaders = patched_get_dataloaders
    except Exception:
        pass
    print(f"[sft_probe_filtered] patched get_dataloaders -> local JSONL loader", flush=True)


def main() -> None:
    jsonl_path = _pop_arg("sft_data_jsonl", "extension/data/sft_probe_filtered_csft.jsonl")
    if jsonl_path and not os.path.isabs(jsonl_path):
        jsonl_path = os.path.join(_REPO_ROOT, jsonl_path)
    _install_local_jsonl_loader(jsonl_path)

    sft_path = os.path.join(_SFT_DIR, "sft.py")
    sys.argv[0] = sft_path
    with open(sft_path) as f:
        code = compile(f.read(), sft_path, "exec")
    exec(code, {"__name__": "__main__", "__file__": sft_path})


if __name__ == "__main__":
    main()
