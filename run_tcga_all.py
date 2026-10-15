"""Run the full TCGA experiment suite; save incremental JSON results."""
import json, os, time
import numpy as np
import torch
torch.set_num_threads(1)
from ahgsl.tcga import run_model_tcga, gene_selection_eval

OUT = "results/tcga_results.json"
GS_OUT = "results/tcga_gene_selection.json"
os.makedirs("results", exist_ok=True)
results = json.load(open(OUT)) if os.path.exists(OUT) else {}
gsres = json.load(open(GS_OUT)) if os.path.exists(GS_OUT) else {}

KINDS = ["ahgsl", "static_hgnn", "gcn", "mlp", "lr",
         "ahgsl_no_aee", "ahgsl_no_ser", "ahgsl_no_gate", "ahgsl_no_stab"]

for fold in range(5):
    for kind in KINDS:
        key = f"{kind}|f{fold}"
        if key in results and (kind != "ahgsl" or f"f{fold}" in gsres):
            continue
        t0 = time.time()
        try:
            if kind == "ahgsl":
                res, model, hist, ctx = run_model_tcga(kind, fold, seed=42,
                                                       return_model=True)
                if f"f{fold}" not in gsres:
                    gs, ranks = gene_selection_eval(model, ctx)
                    gsres[f"f{fold}"] = {"curves": gs,
                                         "top100": {k: v[:100] for k, v in ranks.items()},
                                         "gene_names": ctx["gene_names"].tolist()}
                    json.dump(gsres, open(GS_OUT, "w"))
                if hist is not None:
                    res["refined_frac_hist"] = hist["refined_frac"]
                    res["val_hist"] = hist["val"]
                    res["loss_hist"] = hist["loss"]
            else:
                res = run_model_tcga(kind, fold, seed=42)
            res = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in res.items()}
            results[key] = res
            print(f"[{time.strftime('%H:%M:%S')}] {key}: acc={res['accuracy']:.4f} "
                  f"f1M={res['f1_macro']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[key] = {"error": str(e)}
        json.dump(results, open(OUT, "w"))
print("DONE tcga suite")
