"""Poll Modal until all 5 Phase 2 cache jobs finish.

Phase 2 = the 5 hidden-state cache jobs spawned by launch_expansion_cache.sh.
When all complete, writes a sentinel so the main thread can proceed to
download + analysis.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import modal


JOBS = [
    ("fc-01KSXJR5DH8BB6P1D1J057NCGY", "C_SFT cache (n=500)"),
    ("fc-01KSXJRMEDQFQDX7K3B8WFJXMC", "C_outcome cache (n=500)"),
    ("fc-01KSXJS2W6SRB8KN51QP46D9FJ", "C_outcome_step_30 cache (n=200)"),
    ("fc-01KSXJSGS7X5RXNK19SY3R0M87", "C_outcome_step_60 cache (n=200)"),
    ("fc-01KSXJSZ1W8KRM1KWGEHG03XQR", "C_outcome_step_90 cache (n=200)"),
]
SENTINEL = Path("/tmp/phase2_done.sentinel")


def get_fc_status(fc_id: str) -> str:
    fc = modal.FunctionCall.from_id(fc_id)
    try:
        fc.get(timeout=0)
        return "done"
    except modal.exception.OutputExpiredError:
        return "done"
    except TimeoutError:
        return "pending"
    except Exception as e:
        sys.stderr.write(f"  {fc_id}: error {type(e).__name__}: {e}\n")
        return "failed"


def main() -> None:
    sys.stderr.write(f"[phase2-watchdog] polling {len(JOBS)} Phase 2 cache jobs\n")
    sys.stderr.flush()

    poll_count = 0
    last_statuses: dict[str, str] = {fc_id: "pending" for fc_id, _ in JOBS}
    while True:
        poll_count += 1
        statuses = {fc_id: get_fc_status(fc_id) for fc_id, _ in JOBS}
        for (fc_id, label), s in zip(JOBS, statuses.values()):
            if s != last_statuses[fc_id]:
                sys.stderr.write(f"[phase2-watchdog] poll #{poll_count}: {label} -> {s}\n")
                sys.stderr.flush()
        last_statuses = statuses
        if all(s == "done" for s in statuses.values()):
            sys.stderr.write(f"[phase2-watchdog] ALL DONE after poll #{poll_count}\n")
            sys.stderr.flush()
            SENTINEL.write_text("phase 2 complete\n")
            return
        if any(s == "failed" for s in statuses.values()):
            sys.stderr.write(f"[phase2-watchdog] one or more Phase 2 jobs FAILED\n")
            sys.stderr.flush()
            return
        time.sleep(120)


if __name__ == "__main__":
    main()
