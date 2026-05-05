#!/usr/bin/env python3
from pathlib import Path
import json
import re
import subprocess
import sys

RUN_NAME = 'Stable_diffusion_train_custom_dit_meanflow_250m_30000steps'
CKPT_DIR = Path('/iopsstor/scratch/cscs/dbartaula/experiments_assets/Stable_diffusion_train_custom_dit_meanflow_250m_30000steps/checkpoints')
COMMAND = ['python', 'custom_model_pretraining/evaluate_fid_kid.py', '--approach', 'meanflow', '--checkpoint', '__CKPT_PATH__', '--curvton_test_data_path', '/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate_test', '--triplet_test_data_path', '/iopsstor/scratch/cscs/dbartaula/human_gen/triplet_dataset_backup_1', '--street_tryon_data_path', '/iopsstor/scratch/cscs/dbartaula/human_gen/benchmark_datasets/street_tryon', '--image_size', '512', '--patch_size', '16', '--hidden_size', '1280', '--depth', '9', '--num_heads', '20', '--batch_size', '16', '--num_workers', '8', '--eval_frac_curvton', '0.80', '--eval_frac_triplet', '0.25', '--eval_frac_street', '0.25', '--output_json', '/iopsstor/scratch/cscs/dbartaula/experiments_assets/Stable_diffusion_train_custom_dit_meanflow_250m_30000steps/eval_fid_kid.json']
def _latest_ckpt(ckpt_dir: Path) -> Path:
    candidates = list(ckpt_dir.glob("ckpt_step_*.pt"))
    final_ckpt = ckpt_dir / "ckpt_final.pt"
    if final_ckpt.exists():
        candidates.append(final_ckpt)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    def _step_of(path: Path) -> int:
        if path.name == "ckpt_final.pt":
            return 10**12
        m = re.search(r"ckpt_step_(\d+)\.pt$", path.name)
        return int(m.group(1)) if m else -1

    return max(candidates, key=_step_of)


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
