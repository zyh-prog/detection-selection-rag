"""
over_acceptance.py — Step-2 deliverable: the *mechanism* behind the selection gap.

Claim sharpened from n=2 to n=3 mechanistically-independent realizable selectors:
on CLEAR questions every selector accepts ~the same number of candidates it accepts
on AMBIGUOUS questions — it is blind to ambiguity — so it over-accepts on clear and
pays a large set-F1 penalty there. The failure is therefore a property of the
task+generator (redundant-but-grounded candidates), not of any one selector.

Three realizable selectors, three different mechanisms:
  (A) Self-verification  : prompt-based LLM judgement   -> cached `candidate_valid`
  (B) Lexical grounding  : surface token-recall vs retrieved passages (threshold-free)
  (C) NLI grounding      : DeBERTa entailment vs passages (grounding_selector.py; model)

This file computes (A) and (B) from cache on CPU (no GPU, no model). (C) is the
published F01 selector; its clear/ambiguous keep-rate is reproduced by
`grounding_selector.py` and folded into the write-up separately.

For each cell, on CLEAR questions we report, per selector:
  keptIA      mean # intent-answers accepted ON TOP of the direct answer (ideal = 0)
  keepRate    fraction of generated intent-answers accepted
  F1dir/F1sel set-F1 of {direct} vs {direct + accepted}, and ΔF1 = F1sel - F1dir
  ΔF1 CI      paired bootstrap 95% CI + significance (seed 42) -> penalty is real
and the AMBIGUOUS-question keptIA for the blindness contrast (clear keptIA ~ amb keptIA).

AmbigQA-V3.2 = the 1,902-question cache restricted to the 400 test ids; it also lacks
`candidate_valid`, so selector (A) is n/a there while (B) is still computable.
ASQA has 0 clear questions -> no clear-question analysis.
"""
import sys, re
sys.path.append(".")
import pickle
import numpy as np
import selection_ci as S
from utils.riv_policy import _dedup_semantic

AMB7B_TEST_IDS = set(S.load("data/riv_cache_test.pkl"))
PTEXT = pickle.load(open("data/_f01_passage_text.pkl", "rb"))

STOP = set("a an the of to in on at by for and or is are was were be been being as "
           "with from that this these those it its his her their our your my "
           "who what when where which whom whose how why do does did has have had "
           "s t re ve ll d m".split())
_word = re.compile(r"[a-z0-9]+")


def toks(text):
    return [w for w in _word.findall(str(text).lower()) if w not in STOP and len(w) > 1]


def passages_of(r):
    out = []
    for doc in r.get("retrieved", []):
        t = PTEXT.get(str(doc.get("id")))
        if t:
            out.append(t[0] if isinstance(t, tuple) else t)
    return out or [""]


def lex_grounded(cand, ptok_sets):
    """Threshold-free surface grounding: keep iff every content token of the
    candidate appears in some single retrieved passage (recall == 1.0)."""
    ct = set(toks(cand))
    if not ct:
        return False
    return any(ct <= ps for ps in ptok_sets)


def sv_keep(r):
    v = r.get("candidate_valid")
    if not v:
        return None
    return [a for ok, a in zip(v, S.ia(r)) if ok and str(a).strip()]


def lex_keep(r):
    ps = [set(toks(p)) for p in passages_of(r)]
    return [a for a in S.ia(r) if str(a).strip() and lex_grounded(a, ps)]


def boot(base, treat):
    c = S.bootstrap_confidence_interval(base, treat)
    return c["mean_diff"] * 100, c["ci_lower"] * 100, c["ci_upper"] * 100, c["p_value"], c["significant"]


CELLS = [
    ("AmbigQA 7B",     "data/riv_cache_test.pkl", False),
    ("AmbigQA V3.2*",  "data/_caches_ambigqa1902/riv_cache_test__deepseek-ai_DeepSeek-V3.2.pkl", True),
    ("ASQA 7B",        "data/asqa/riv_cache_test__qwen2.5-instruct.pkl", False),
    ("ASQA V3.2",      "data/asqa/riv_cache_test__deepseek-ai_DeepSeek-V3.2.pkl", False),
    ("SituatedQA 7B",  "data/situatedqa/riv_cache_test__qwen2.5-instruct.pkl", False),
    ("SituatedQA V3.2","data/situatedqa/riv_cache_test__deepseek-ai_DeepSeek-V3.2.pkl", False),
]

SELECTORS = [("self-verif", sv_keep), ("lexical-grnd", lex_keep)]


def seg_stats(rows, keepfn):
    """Per-row stats on a question segment (clear or ambiguous)."""
    keptn, kr, f1d, f1s = [], [], [], []
    for r in rows:
        g, d, ia = S.gt(r), S.direct(r), S.ia(r)
        kept = keepfn(r)
        if kept is None:
            return None
        kset = _dedup_semantic([d] + kept) if d else (_dedup_semantic(kept) or [d])
        keptn.append(len(kept))
        kr.append(len(kept) / max(len(ia), 1))
        f1d.append(S.compute_ambigqa_f1(d, g))
        f1s.append(S.f1(kset, g))
    return keptn, kr, f1d, f1s


def main():
    for lab, path, restrict in CELLS:
        recs = S.load(path)
        keys = sorted(set(recs) & AMB7B_TEST_IDS) if restrict else sorted(recs)
        clr = [recs[k] for k in keys if not bool(recs[k].get("is_ambiguous", False))]
        amb = [recs[k] for k in keys if bool(recs[k].get("is_ambiguous", False))]
        n = len(keys); cfrac = len(clr) / n; afrac = len(amb) / n
        print("=" * 104)
        print(f"[{lab}]  n={n}  clear={len(clr)} ({cfrac*100:.0f}%)  ambiguous={len(amb)} ({afrac*100:.0f}%)"
              + ("   (0 clear -> over-acceptance-on-clear undefined)" if not clr else ""))
        if not clr:
            continue
        for sname, kf in SELECTORS:
            cs = seg_stats(clr, kf)
            if cs is None:
                print(f"   {sname:<13} (no candidate_valid in cache -> n/a)")
                continue
            kn_c, kr_c, f1d_c, f1s_c = cs
            as_ = seg_stats(amb, kf)
            kn_a, _, f1d_a, f1s_a = as_
            d, ci_l, ci_u, p, sig = boot(f1d_c, f1s_c)              # clear-question penalty + CI
            df1_c = np.mean(f1s_c) - np.mean(f1d_c)                  # clear ΔF1 (fraction)
            df1_a = np.mean(f1s_a) - np.mean(f1d_a)                  # ambiguous ΔF1 (fraction)
            drag = cfrac * df1_c * 100                               # clear contribution to overall ΔF1
            lift = afrac * df1_a * 100                               # ambiguous contribution
            net = drag + lift                                        # = overall ΔF1 (exact identity)
            print(f"   {sname:<13} CLEAR: keptIA={np.mean(kn_c):.2f}  keepRate={np.mean(kr_c)*100:4.1f}%  "
                  f"F1dir={np.mean(f1d_c)*100:5.1f} F1sel={np.mean(f1s_c)*100:5.1f}  "
                  f"ΔF1={d:+6.2f} CI[{ci_l:+.2f},{ci_u:+.2f}] {'*' if sig else 'n.s.'}"
                  f"   | AMB keptIA={np.mean(kn_a):.2f} blind-gap={np.mean(kn_c)-np.mean(kn_a):+.2f}")
            print(f"   {'':13} DECOMP: drag(clear)={drag:+5.2f}  +  lift(amb)={lift:+5.2f}  "
                  f"=  net overall ΔF1={net:+5.2f}  -> selection {'HURTS' if net < 0 else 'helps'} overall")

    print("\nREAD:")
    print(" blind-gap = clear keptIA - ambiguous keptIA. ~0 => selector accepts the SAME #candidates")
    print("   on clear as on ambiguous questions = blind to ambiguity (the root cause of over-accept).")
    print(" ΔF1<0, CI excluding 0 => over-acceptance significantly hurts on clear questions.")
    print(" DECOMP is an exact identity: overall ΔF1 = clear_frac*ΔF1_clear + amb_frac*ΔF1_amb.")
    print("   The clear-question 'drag' (always <0) vs ambiguous 'lift' (>0); whichever dominates")
    print("   sets the sign. High clear-fraction (AmbigQA 40%) => drag wins => realizable selection")
    print("   NETS NEGATIVE; zero clear-fraction (ASQA) => no drag => selection helps. Same law,")
    print("   two mechanistically-independent selectors (LLM self-verification + surface grounding).")


if __name__ == "__main__":
    main()
