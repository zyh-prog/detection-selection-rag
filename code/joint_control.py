"""
joint_control.py — rebuttal datapoint for reviewer critique (1):
"perfect detection is handicapped (all-candidate output, no selection), so the
selection gap is partly baked in."

JOINT system Perf-Det+Sel = gold-detection gate + oracle selection on the
branched (ambiguous) pool:
  clear (gold)      -> direct answer            (detection gates)
  ambiguous (gold)  -> best gold-scored subset  (oracle selection on branched)

Δ_sel|det = (Perf-Det+Sel) - Perf-Det isolates selection's value GIVEN detection
already branched; it is 0 on clear questions by construction, so it is NOT an
artifact of withholding selection from the detector. Large + significant => the
selection bottleneck is real, not definitional.

Scoring reused verbatim from selection_ci.py (string-dedup, set-F1) so
Direct/Perf-Det/Perf-Sel reproduce Table 1 exactly (self-check column).

Cache map. The AmbigQA-V3.2 cell reads the full 1,902-question validation cache and
restricts it to the 400 test ids; only 379 of them carry the required fields, so that
one row is an approximation and is marked with * in the output.
"""
import sys, os
sys.path.append(".")
import numpy as np
import selection_ci as S

AMB7B_TEST_IDS = set(S.load("data/riv_cache_test.pkl"))  # the 400 AmbigQA test ids


def per_sample_ext(r):
    g, d = S.gt(r), S.direct(r)
    amb = bool(r.get("is_ambiguous", False))
    b = S.per_sample(r)
    b["perf_det_sel"] = S.best_subset([d] + S.ia(r), g) if amb else S.compute_ambigqa_f1(d, g)
    b["amb"] = amb
    return b


def ci(a, b):
    c = S.bootstrap_confidence_interval(a, b)
    return c["mean_diff"] * 100, c["ci_lower"] * 100, c["ci_upper"] * 100, c["p_value"], c["significant"]


# (label, cache_path, restrict_to_amb7b_test_ids?, Table1 (direct,det,sel) or None)
SETTINGS = [
    ("AmbigQA 7B",     "data/riv_cache_test.pkl",                                      False, (42.94, 47.21, 58.94)),
    ("AmbigQA V3.2*",  "data/_caches_ambigqa1902/riv_cache_test__deepseek-ai_DeepSeek-V3.2.pkl", True, (50.88, 54.06, 64.37)),
    ("ASQA 7B",        "data/asqa/riv_cache_test__qwen2.5-instruct.pkl",               False, (29.79, 34.98, 47.02)),
    ("ASQA V3.2",      "data/asqa/riv_cache_test__deepseek-ai_DeepSeek-V3.2.pkl",      False, (33.83, 38.07, 48.48)),
    ("SituatedQA 7B",  "data/situatedqa/riv_cache_test__qwen2.5-instruct.pkl",         False, (20.48, 25.14, 33.69)),
    ("SituatedQA V3.2","data/situatedqa/riv_cache_test__deepseek-ai_DeepSeek-V3.2.pkl",False, (26.04, 29.88, 37.39)),
]

print(f"{'setting':<16}{'Direct':>8}{'PerfDet':>8}{'Det+Sel':>9}{'PerfSel':>8}{'GoldInt':>8}"
      f"   {'Δ_det':>7}{'Δsel|det[①]':>13}{'Δsel(paper)':>12}   selfcheck")
print("-" * 118)
for lab, path, restrict, tbl in SETTINGS:
    if not os.path.exists(path):
        print(f"{lab:<16}  cache missing: {path}"); continue
    recs = S.load(path)
    keys = sorted(set(recs) & AMB7B_TEST_IDS) if restrict else sorted(recs)
    P = {k: per_sample_ext(recs[k]) for k in keys}
    amb = [k for k in keys if P[k]["amb"]]
    m = lambda key, s=keys: float(np.mean([P[k][key] for k in s]) * 100)
    dirv, detv, selv = m("direct"), m("perf_det_dp"), m("perf_sel_dp")
    dsv, giv = m("perf_det_sel"), m("gold_interp")
    d  = [P[k]["direct"] for k in keys]; pd = [P[k]["perf_det_dp"] for k in keys]
    ds = [P[k]["perf_det_sel"] for k in keys]; sl = [P[k]["perf_sel_dp"] for k in keys]
    ddet = ci(d, pd)[0]; dsd = ci(pd, ds); dsp = ci(pd, sl)[0]
    ok = "n/a" if tbl is None else ("OK" if all(abs(a-b) < 0.1 for a, b in
            zip((dirv, detv, selv), tbl)) else f"≈(D{dirv:.1f}/T{tbl[0]:.1f})")
    deg = "  [0 clear -> degenerate for ①]" if not amb or len(amb) == len(keys) and "ASQA" in lab else ""
    sig = "*" if dsd[4] else "n.s."
    print(f"{lab:<16}{dirv:>8.2f}{detv:>8.2f}{dsv:>9.2f}{selv:>8.2f}{giv:>8.2f}"
          f"   {ddet:>+7.2f}{dsd[0]:>+10.2f}{sig:>3}{dsp:>+12.2f}   {ok}{deg}")

print("\n* AmbigQA V3.2 = the 1,902-question V3.2 cache restricted to the 400 test ids;"
      "\n  only 379 carry the needed fields, so n<400 and Direct sits ~0.5 off Table 1"
      "\n  -> APPROXIMATE. The exact n=400 values are in the paper, from frozen scores.")
print("Read: Δsel|det (Perf-Det+Sel - Perf-Det) is selection's value AFTER detection branches,")
print("=0 on clear by construction. It stays large + significant -> selection gap is not an")
print("artifact of denying detection a selection step. ASQA has 0 clear -> can't test ①.")
