"""Run the full yeast experiment suite; save incremental JSON results."""
import json, os, time, sys
import numpy as np
import torch
torch.set_num_threads(1)
from ahgsl.train import run_model_yeast

# optional CLI filter: run only these datasets (comma-separated)
ONLY = sys.argv[1].split(",") if len(sys.argv) > 1 else None
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/yeast_results.json"
os.makedirs("results", exist_ok=True)
results = {}
if os.path.exists(OUT):
    results = json.load(open(OUT))

DATASETS = ["cellcycle", "church", "derisi", "eisen", "yeast14"]
if ONLY:
    DATASETS = [d for d in DATASETS if d in ONLY]
MAIN = ["ahgsl", "static_hgnn", "gcn", "mlp", "br_lr"]
ABL = ["ahgsl_no_aee", "ahgsl_no_ser", "ahgsl_no_gate", "ahgsl_no_stab"]
SEEDS_MAIN = [42, 1, 7]          # 3 seeds for all main models
SEEDS_ABL = [42, 1]              # two seeds for ablations

jobs = []
for d in DATASETS:
    for k in MAIN:
        seeds = SEEDS_MAIN if k not in ("br_lr",) else [42]  # LR deterministic
        for s in seeds:
            jobs.append((d, k, s))
    for k in ABL:
        for s in SEEDS_ABL:
            jobs.append((d, k, s))

for d, k, s in jobs:
    key = f"{d}|{k}|{s}"
    if key in results:
        continue
    t0 = time.time()
    try:
        out = run_model_yeast(d, k, seed=s)
        hist = None
        if k.startswith(("ahgsl", "static")):
            pass
        out = {kk: (float(v) if isinstance(v, (int, float, np.floating)) else v)
               for kk, v in out.items()}
        results[key] = out
        print(f"[{time.strftime('%H:%M:%S')}] {key}: f1={out['f1_micro']:.4f} "
              f"AP={out['avg_precision_micro']:.4f} auc={out['auc_micro']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"[FAIL] {key}: {e}", flush=True)
        results[key] = {"error": str(e)}
    json.dump(results, open(OUT, "w"), indent=1)
print("DONE yeast suite")
