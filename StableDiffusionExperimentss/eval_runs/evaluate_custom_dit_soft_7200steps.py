#!/usr/bin/env python3
from pathlib import Path
import json
import re
import subprocess
import sys

RUN_NAME = 'train_custom_dit_soft_7200steps'
CKPT_DIR = Path('/iopsstor/scratch/cscs/dbartaula/custom_dit_assets/train_custom_dit_soft_7200steps/checkpoints')
COMMAND = ['python', 'custom_model_pretraining/evaluate.py', '--checkpoint', '__CKPT_PATH__', '--curvton_test_data_path', '/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate_test', '--batch_size', '16', '--num_workers', '8', '--eval_frac_curvton', '0.80', '--eval_frac_curvton_overall', '0.25']
def _latest_ckpt(ckpt_dir: Path) -> Path:
    final_ckpt = ckpt_dir / "ckpt_final.pt"
    if final_ckpt.exists():
        return final_ckpt
    raise FileNotFoundError(f"No final checkpoint found in {ckpt_dir} (expected ckpt_final.pt)")


def _print_output_json_if_available(cmd_tokens):
    if "--output_json" not in cmd_tokens:
        return
    try:
        idx = cmd_tokens.index("--output_json")
        out_path = Path(cmd_tokens[idx + 1])
    except (ValueError, IndexError):
        return

    if not out_path.exists():
        return

    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
        print("\nMetrics summary (from output_json):")
        print(json.dumps(data, indent=2))
    except Exception as exc:
        print(f"WARNING: could not read output JSON {out_path}: {exc}")


def main() -> int:
    extra_args = sys.argv[1:]
    ckpt_path = None

    if extra_args:
        if extra_args[0].endswith(".pt"):
            ckpt_path = Path(extra_args[0])
            extra_args = extra_args[1:]
        elif "--checkpoint" in extra_args:
            i = extra_args.index("--checkpoint")
            if i + 1 < len(extra_args):
                ckpt_path = Path(extra_args[i + 1])
                del extra_args[i:i + 2]

    if ckpt_path is None:
        if not CKPT_DIR.exists():
            print(f"Evaluating run: {RUN_NAME}")
            print(f"Checkpoint directory not found: {CKPT_DIR}")
            print("Stopping evaluation (no fallback allowed).", file=sys.stderr)
            return 1

        try:
            ckpt_path = _latest_ckpt(CKPT_DIR)
        except FileNotFoundError:
            print(f"Evaluating run: {RUN_NAME}")
            print(f"No checkpoint found in: {CKPT_DIR}")
            print("Stopping evaluation (no fallback allowed).", file=sys.stderr)
            return 1

    cmd = [str(ckpt_path) if t == "__CKPT_PATH__" else t for t in COMMAND]

    if extra_args:
        print("Forwarding extra CLI args:", " ".join(extra_args))
        cmd = cmd + extra_args

    print(f"Evaluating run: {RUN_NAME}")
    print(f"Using checkpoint: {ckpt_path}")
    print("Command:", " ".join(cmd))

    result = subprocess.run(cmd)
    if result.returncode == 0:
        _print_output_json_if_available(cmd)
    return result.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)