# train_runs

Centralized training runners generated from existing `train_*.sh` scripts.

Run one script:

```bash
python train_runs/<runner>.py
```

Run all scripts:

```bash
for f in train_runs/*.py; do python "$f"; done
```
