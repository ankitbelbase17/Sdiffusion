"""
Exhaustive validation of both triplet datasets (train + test).

Iterates ALL batches of every loader and classifies each batch as:
  GOOD   – collate returned a valid batch with no padding needed
  PADDED – collate had to replace some None samples (logged warning)
  FAILED – collate returned None (entire batch was invalid)

Summary printed at the end.
"""

import sys, os, logging, time

sys.path.insert(0, os.path.dirname(__file__))

try:
    from config import TRIPLET_TRAIN_PATH, TRIPLET_TEST_PATH
except Exception:
    TRIPLET_TRAIN_PATH = "/iopsstor/scratch/cscs/dbartaula/human_gen/triplet_dataset_backup_1"
    TRIPLET_TEST_PATH = "/iopsstor/scratch/cscs/dbartaula/human_gen/triplet_dataset_backup_1"
from utils import get_triplet_train_loader, get_triplet_test_dataloaders, collate_fn

# ── Intercept collate warnings to detect padded batches ──────
_padded_flag = False
_orig_warning = None

class _CollatePadDetector(logging.Handler):
    """Watches for the 'bad samples — padding batch' warning from collate_fn."""
    def emit(self, record):
        global _padded_flag
        if "padding batch" in record.getMessage():
            _padded_flag = True

_handler = _CollatePadDetector()
_handler.setLevel(logging.WARNING)
logging.getLogger("utils").addHandler(_handler)

BATCH_SIZE   = 4
NUM_WORKERS  = 2
IMAGE_SIZE   = 512


def validate_loader(name: str, loader):
    """Run through every batch and classify as good / padded / failed."""
    global _padded_flag

    total   = len(loader)
    good    = 0
    padded  = 0
    failed  = 0
    failed_indices = []

    print(f"\n{'='*70}")
    print(f"  {name}  —  {total} batches")
    print(f"{'='*70}")

    t0 = time.time()
    for i, batch in enumerate(loader):
        if batch is None:
            failed += 1
            failed_indices.append(i)
            print(f"  [FAILED] batch {i+1}/{total}")
        elif _padded_flag:
            padded += 1
            print(f"  [PADDED] batch {i+1}/{total}")
        else:
            good += 1

        _padded_flag = False  # reset for next batch

        # Progress every 100 batches
        if (i + 1) % 100 == 0:
            print(f"  ... processed {i+1}/{total} batches")

    elapsed = time.time() - t0
    print(f"\n  --- {name} Summary ---")
    print(f"  Total batches : {total}")
    print(f"  GOOD          : {good}")
    print(f"  PADDED        : {padded}")
    print(f"  FAILED (None) : {failed}")
    if failed_indices:
        show = failed_indices[:20]
        print(f"  Failed batch indices: {show}{'...' if len(failed_indices)>20 else ''}")
    print(f"  Time          : {elapsed:.1f}s")
    return {"name": name, "total": total, "good": good, "padded": padded, "failed": failed}


# ── Main ─────────────────────────────────────────────────────
results = []

# 1) Triplet TRAIN dataset (triplet_dataset_backup_1_1 = Phase 2 training data)
print(f"\n>>> Triplet TRAIN dataset: {TRIPLET_TRAIN_PATH}")
train_loader, _ = get_triplet_train_loader(
    root_dir=TRIPLET_TRAIN_PATH,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    size=IMAGE_SIZE,
    world_size=1,
    rank=0,
)
results.append(validate_loader("TRIPLET TRAIN (combined)", train_loader))

# 2) Triplet TEST dataset (triplet_dataset_backup_1 = evaluation data, per-subset loaders)
print(f"\n>>> Triplet TEST dataset: {TRIPLET_TEST_PATH}")
test_loaders = get_triplet_test_dataloaders(
    root_dir=TRIPLET_TEST_PATH,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    size=IMAGE_SIZE,
)
for key, loader in test_loaders.items():
    results.append(validate_loader(f"TRIPLET TEST / {key}", loader))


# ── Final report ─────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  FINAL REPORT")
print(f"{'='*70}")
print(f"  {'Dataset':<35} {'Total':>6} {'Good':>6} {'Padded':>7} {'Failed':>7}")
print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*7} {'-'*7}")
for r in results:
    print(f"  {r['name']:<35} {r['total']:>6} {r['good']:>6} {r['padded']:>7} {r['failed']:>7}")

all_good = all(r["failed"] == 0 and r["padded"] == 0 for r in results)
print(f"\n  Overall: {'ALL CLEAN' if all_good else 'ISSUES FOUND — see above'}")
print(f"{'='*70}")
