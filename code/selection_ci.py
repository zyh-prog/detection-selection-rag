"""
selection_ci.py — adds the two CIs the spine sentence needs, on top of the
four-point ladder:
  - selection : perf_sel_dp - perf_det_dp  (value of picking the right subset given perfect detection)
  - contrast  : selection - detection       (whether selection statistically beats detection; the "~3x" evidence)
Scoring reuses the ladder verbatim, so the four-point table should match it (self-check).
Usage (same four args as the ladder):
  python selection_ci.py \
    "Qwen2.5-7B=data/riv_cache_dev.pkl" \
    "Qwen3-14B=data/riv_cache_dev__Qwen_Qwen3-14B.pkl" \
    "Qwen3-32B=data/riv_cache_dev__Qwen_Qwen3-32B.pkl" \
    "DeepSeek-V3.2=data/riv_cache_dev__deepseek-ai_DeepSeek-V3.2.pkl"
"""
import sys, pickle, itertools
import numpy as np
sys.path.append(".")
from utils.metrics import compute_ambigqa_f1, compute_ambigqa_f1_multi, bootstrap_confidence_interval
from utils.riv_policy import _dedup_semantic

# --- Naming vs. paper terminology (LNCS submission) ---
# Internal metric names follow the paper's terms; persisted cache keys are left
# UNCHANGED (frozen .pkl): oracle_intent_answers / oracle_answers / intent_answers.
#   gold_interp = Gold interpretation | perf_sel = Perfect selection | perf_det = Perfect detection


# ---- load / accessors / f1 / best_subset / per_sample: reused verbatim from ladder.py ----
def load(path):
    obj = pickle.load(open(path, "rb"))
    if isinstance(obj, dict) and "records" in obj:
        recs = obj["records"]
    elif isinstance(obj, dict):
        recs = {k: v for k, v in obj.items() if not str(k).startswith("_") and isinstance(v, dict)}
    else:
        recs = {str(r.get("id", i)): r for i, r in enumerate(obj)}
    return {str(k): v for k, v in recs.items()}


def gt(r):     return r.get("ground_truth_answers") or r.get("gt_answers") or []
def direct(r): return r.get("direct_answer") or (r.get("ans_texts") or [""])[0] or ""
def ia(r):     return r.get("intent_answers") or []
def oa(r):     return r.get("oracle_intent_answers") or r.get("oracle_answers") or []


def f1(preds, g):
    preds = [str(p).strip() for p in preds if str(p).strip()]
    if not preds:
        return 0.0
    return compute_ambigqa_f1(preds[0], g) if len(preds) == 1 else compute_ambigqa_f1_multi(preds, g)


def best_subset(cands, g, max_n=5):
    cands = _dedup_semantic([x for x in cands if str(x).strip()])[:max_n]
    best = 0.0
    for k in range(1, len(cands) + 1):
        for idx in itertools.combinations(range(len(cands)), k):
            best = max(best, f1([cands[i] for i in idx], g))
    return best


def per_sample(r):
    g, d = gt(r), direct(r)
    amb = bool(r.get("is_ambiguous", False))
    return dict(direct=compute_ambigqa_f1(d, g),
                gold_interp=f1(oa(r), g),
                perf_sel_dp=best_subset([d] + ia(r), g),
                perf_det_dp=(f1(_dedup_semantic([d] + ia(r)) or [d], g) if amb else compute_ambigqa_f1(d, g)))
# ---- end reuse ----


def fmt(ci):
    return (f"Δ={ci['mean_diff']*100:+6.2f}%  "
            f"CI[{ci['ci_lower']*100:+.2f}, {ci['ci_upper']*100:+.2f}]  "
            f"p={ci['p_value']:.3f}  {'*' if ci['significant'] else 'n.s.'}")


def main():
    items = [a.split("=", 1) for a in sys.argv[1:]]
    if len(items) < 1:
        raise SystemExit('usage: python selection_ci.py "label=path" ["label=path" ...]')
    caches = {lab: load(p) for lab, p in items}
    common = sorted(set.intersection(*[set(c) for c in caches.values()]))
    if not common:
        raise SystemExit("no common ids; cannot align for comparison.")
    print(f"aligned common ids: {len(common)}  (per-cache n: " +
          ", ".join(f"{lab}:{len(c)}" for lab, c in caches.items()) + ")\n")

    P = {lab: {k: per_sample(c[k]) for k in common} for lab, c in caches.items()}
    keys = ["direct", "gold_interp", "perf_sel_dp", "perf_det_dp"]

    print(f"{'model':<16}" + "".join(f"{k:>16}" for k in keys) + "   (should match the ladder)")
    print("-" * (16 + 16 * len(keys)))
    for lab, _ in items:
        row = P[lab]
        print(f"{lab:<16}" + "".join(f"{np.mean([row[k][key] for k in common])*100:>16.2f}" for key in keys))

    def col(row, key): return [row[k][key] for k in common]

    print("\n[A] load-bearing deltas (X - direct; paired bootstrap 95% CI; should match the ladder)")
    print("-" * 78)
    for label, hi in [("realizable  gold_interp - direct", "gold_interp"),
                      ("candidate   perf_sel_dp - direct", "perf_sel_dp"),
                      ("detection   perf_det_dp - direct", "perf_det_dp")]:
        print(label)
        for lab, _ in items:
            print(f"  {lab:<16} {fmt(bootstrap_confidence_interval(col(P[lab], 'direct'), col(P[lab], hi)))}")

    print("\n[B] selection vs detection decomposition (spine evidence; paired bootstrap 95% CI)")
    print("-" * 78)
    for lab, _ in items:
        row = P[lab]
        d, pg, go = col(row, "direct"), col(row, "perf_det_dp"), col(row, "perf_sel_dp")
        sel = [g - p for g, p in zip(go, pg)]     # selection_i
        det = [p - x for p, x in zip(pg, d)]      # detection_i
        sel_ci  = bootstrap_confidence_interval(pg, go)    # Δ = perf_sel - perf_det (selection)
        cont_ci = bootstrap_confidence_interval(det, sel)  # Δ = selection - detection (contrast)
        print(f"  [{lab}]")
        print(f"     selection  perf_sel_dp - perf_det_dp   {fmt(sel_ci)}")
        print(f"     contrast   selection - detection       {fmt(cont_ci)}")

    print("\nnote: if the [B] contrast CI is > 0 for all models, 'selection statistically beats "
          "detection' holds; use that (rather than non-overlapping delta CIs) to support the spine sentence.")


if __name__ == "__main__":
    main()
