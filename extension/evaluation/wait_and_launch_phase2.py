"""Poll Modal until all 5 Phase 1 rollout jobs finish, then trigger Phase 2.

Phase 1 = the 5 vLLM rollout jobs spawned by launch_expansion_rollouts.sh.
Phase 2 = the 5 hidden-state cache jobs in launch_expansion_cache.sh.

Polls each `modal.FunctionCall` with a non-blocking `.get(timeout=0)` and waits
~60s between polls. Writes a single line per status change to stderr so the
runner can be backgrounded.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import modal


JOBS = [
    ("fc-01KSXGZM4WJW4H6ESAKKXPJG8R", "C_SFT n=500"),
    ("fc-01KSXH02Z4Z9X06Y9FQJFJ6ZE5", "C_outcome n=500"),
    ("fc-01KSXH0HXYNER98HND67MRQJ3C", "C_outcome_step_30 n=200"),
    ("fc-01KSXH117SGQCTSVDC39AAP68C", "C_outcome_step_60 n=200"),
    ("fc-01KSXH1HKEPC7FXHRP550032MG", "C_outcome_step_90 n=200"),
]
PHASE2 = Path(__file__).resolve().parent.parent / "probe" / "launch_expansion_cache.sh"
SENTINEL = Path("/tmp/phase2_launched.sentinel")


def get_fc_status(fc_id: str) -> str:
    """Return one of: 'pending', 'done', 'failed'."""
    fc = modal.FunctionCall.from_id(fc_id)
    try:
        fc.get(timeout=0)
        return "done"
    except modal.exception.OutputExpiredError:
        return "done"  # output expired = already completed long ago
    except TimeoutError:
        return "pending"
    except Exception as e:
        sys.stderr.write(f"  {fc_id}: error {type(e).__name__}: {e}\n")
        return "failed"


def main() -> None:
    sys.stderr.write(f"[watchdog] polling {len(JOBS)} Phase 1 jobs, will trigger Phase 2 when all done\n")
    sys.stderr.write(f"[watchdog] phase 2 launcher: {PHASE2}\n")
    sys.stderr.flush()

    poll_count = 0
    last_statuses: dict[str, str] = {fc_id: "pending" for fc_id, _ in JOBS}
    while True:
        poll_count += 1
        any_failed = False
        statuses: dict[str, str] = {}
        for fc_id, label in JOBS:
            status = get_fc_status(fc_id)
            statuses[fc_id] = status
            if status != last_statuses[fc_id]:
                sys.stderr.write(f"[watchdog] poll #{poll_count}: {label} -> {status}\n")
                sys.stderr.flush()
            if status == "failed":
                any_failed = True
        last_statuses = statuses

        if all(s == "done" for s in statuses.values()):
            sys.stderr.write(f"[watchdog] ALL DONE after poll #{poll_count}; launching Phase 2\n")
            sys.stderr.flush()
            break
        if any_failed:
            sys.stderr.write(f"[watchdog] one or more Phase 1 jobs FAILED; aborting Phase 2 launch\n")
            sys.stderr.flush()
            return

        time.sleep(60)

    # Launch Phase 2.
    env = os.environ.copy()
    # Source .env explicitly because this is launched from a non-login bash.
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")

    sys.stderr.write(f"[watchdog] running {PHASE2}\n")
    sys.stderr.flush()
    result = subprocess.run(["bash", str(PHASE2)], env=env, capture_output=True, text=True)
    sys.stderr.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.stderr.write(f"[watchdog] phase 2 launcher exited with rc={result.returncode}\n")
    SENTINEL.write_text(f"rc={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}\n")
    sys.stderr.flush()


if __name__ == "__main__":
    main()
