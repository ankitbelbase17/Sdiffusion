import glob
import re
from pathlib import Path

eval_runs = Path("eval_runs")

new_func = '''def _latest_ckpt(ckpt_dir: Path) -> Path:
    final_ckpt = ckpt_dir / "ckpt_final.pt"
    if final_ckpt.exists():
        return final_ckpt
    raise FileNotFoundError(f"No final checkpoint found in {ckpt_dir} (expected ckpt_final.pt)")'''

pattern = re.compile(r'def _latest_ckpt\(ckpt_dir: Path\) -> Path:.*?return candidates\[-1\]', re.DOTALL)

for p in eval_runs.glob("*.py"):
    # Skip the ones strictly requested to use 6000 or 12000
    name = p.name
    if any(x in name for x in ["easy_only", "medium_only", "hard_only", "easy_hard", "medium_hard"]):
        continue
    
    content = p.read_text(encoding="utf-8")
    
    if pattern.search(content):
        new_content = pattern.sub(new_func, content)
        p.write_text(new_content, encoding="utf-8")
        print(f"Updated {p.name}")
