#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGET_SCRIPT = ROOT / 'scripts_catvton/percentage/train_soft_curriculum_catvton_50pct.sh'


def main() -> int:
    if not TARGET_SCRIPT.exists():
        print(f"ERROR: training script not found: {TARGET_SCRIPT}", file=sys.stderr)
        return 1

    cmd = ["bash", str(TARGET_SCRIPT)]
    print(f"Running training script: {TARGET_SCRIPT}")
    print("Command:", " ".join(cmd))
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
