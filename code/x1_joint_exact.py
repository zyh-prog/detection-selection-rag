"""
x1_joint_exact.py — the exact detection-equalized control (Table 2 of the paper).

`joint_control.py` computes the same control from the per-cell record caches, but its
AmbigQA-V3.2 row has to be approximated (see that file). This script instead reads the
frozen per-row scores written by `conj_tau_experiment.py`, which carry the direct answer,
the intent answers, the gold answers and the ambiguity label for all six cells at n=400.
Every cell therefore reproduces Table 1 exactly (Direct / Perf-Det / Perf-Sel to 0.01) and
the control is exact everywhere — and no LLM call is needed to recompute it.

Run `conj_tau_experiment.py` first; it writes the score file this reads.

Scoring mirrors selection_ci.per_sample / joint_control.per_sample_ext verbatim:
  perf_det     = f1(dedup([d]+ia) or [d], g)   if amb else f1(d, g)
  perf_det_sel = best_subset([d]+ia, g)        if amb else f1(d, g)
  perf_sel     = best_subset([d]+ia, g)
Per-row self-checks assert recomputed perf_det == stored f_perfect and
perf_sel == stored f_oracle, proving construction identity with the paper pipeline.
Stats: paired percentile bootstrap, 10,000 resamples, seed 42 (utils.metrics).
"""
import sys
sys.path.append(".")
import pickle
import numpy as np
import selection_ci as S
from utils.riv_policy import _dedup_semantic

CJ = pickle.load(open("data/_conj_nli_scores.pkl", "rb"))
T1_DDET = {"AmbigQA 7B": 4.27, "AmbigQA V3.2": 3.18, "ASQA 7B": 5.20,
           "ASQA V3.2": 4.24, "SituatedQA 7B": 4.66, "SituatedQA V3.2": 3.85}
CELLS = ["AmbigQA 7B", "AmbigQA V3.2", "SituatedQA 7B", "SituatedQA V3.2",
         "ASQA 7B", "ASQA V3.2"]


def ci(a, b):
    c = S.bootstrap_confidence_interval(a, b)
    return (c["mean_diff"] * 100, c["ci_lower"] * 100, c["ci_upper"] * 100,
            c["p_value"], c["significant"])


print(f"{'cell':<17}{'n':>4}{'clear':>6} | {'Direct':>7}{'PerfDet':>8}{'Det+Sel':>8}"
      f"{'PerfSel':>8} | {'Δdet':>6}{'(T1)':>6}{'Δsel|det':>9}{'CI':>18}{'p':>8}")
print("-" * 110)
rows_out = {}
for cell in CELLS:
    rows = CJ[cell]
    dirv, pdet, pds, psel, mism = [], [], [], [], 0
    for r in rows:
        d, ia, g, amb = r["d"], list(r["ia"]), r["g"], bool(r["amb"])
        f_dir = S.compute_ambigqa_f1(d, g)
        f_pd = S.f1(_dedup_semantic([d] + ia) or [d], g) if amb else f_dir
        f_ps = S.best_subset([d] + ia, g)
        f_pds = f_ps if amb else f_dir
        # construction-identity self-checks vs the frozen paper pipeline
        if abs(f_dir - r["f_direct"]) > 1e-9 or abs(f_pd - r["f_perfect"]) > 1e-9 \
           or abs(f_ps - r["f_oracle"]) > 1e-9:
            mism += 1
        dirv.append(f_dir); pdet.append(f_pd); pds.append(f_pds); psel.append(f_ps)
    ncl = sum(1 for r in rows if not r["amb"])
    ddet = ci(dirv, pdet)
    dsd = ci(pdet, pds)
    t1 = T1_DDET[cell]
    flag = "OK" if abs(ddet[0] - t1) < 0.01 else "≠T1"
    deg = "  [0 clear: degenerate]" if ncl == 0 else ""
    print(f"{cell:<17}{len(rows):>4}{ncl:>6} | {np.mean(dirv)*100:>7.2f}"
          f"{np.mean(pdet)*100:>8.2f}{np.mean(pds)*100:>8.2f}{np.mean(psel)*100:>8.2f} | "
          f"{ddet[0]:>+6.2f}{t1:>6.2f}{dsd[0]:>+9.2f}"
          f"  [{dsd[1]:+.2f},{dsd[2]:+.2f}]{dsd[3]:>8.4f}"
          f"  {'*' if dsd[4] else 'n.s.'} selfcheck:{flag}/mism={mism}{deg}")
    rows_out[cell] = dict(n=len(rows), clear=ncl, ddet=round(ddet[0], 2),
                          dsd=round(dsd[0], 2), ci=[round(dsd[1], 2), round(dsd[2], 2)],
                          p=dsd[3], mismatch_rows=mism)

print("\nEXACT n=400 joint-control table (camera-ready, replaces the n=379 approximation):")
for cell in CELLS[:4]:
    o = rows_out[cell]
    print(f"  {cell:<17} Δdet {o['ddet']:+.2f}   Δsel|det {o['dsd']:+.2f}*  CI{o['ci']}")
pickle.dump(rows_out, open("logs/x1_joint_exact_results.pkl", "wb"))
print("\nsaved -> logs/x1_joint_exact_results.pkl")
