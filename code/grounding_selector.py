"""grounding_selector.py — an evidence-grounded (NLI entailment) selector.

A second, mechanistically *independent* deployable selector to test whether the
AmbigQA failure of prompt-based self-verification is intrinsic to realizable
selection or specific to that one mechanism.

For each candidate answer we form the hypothesis
    "<question> The answer is <candidate>."
and run an NLI model with each of the record's retrieved passages as premise.
A candidate is KEPT iff, on its best-supporting passage, the top NLI label is
*entailment* (a threshold-free, no-tuning decision rule — nothing is fit on the
data). The kept set is deduplicated; empty sets fall back to the direct answer.

We report set-F1 of the grounded selector against Direct and Perfect detection
(paired bootstrap, 10k resamples, seed 42), plus realized headroom
(grounded - direct)/(perf_sel - direct), and keep-rate on clear vs ambiguous
questions. Reuses the same set-F1 / bootstrap utilities as the ladder/selector scripts.

Usage:
    python grounding_selector.py
    python grounding_selector.py --tau 0.5   # optional entail-prob threshold instead of argmax
"""
import sys, pickle, itertools, argparse
import numpy as np
sys.path.append(".")
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.metrics import compute_ambigqa_f1, compute_ambigqa_f1_multi, bootstrap_confidence_interval
from utils.riv_policy import _dedup_semantic

# Naming vs. paper: perf_sel=Perfect selection, perf_det=Perfect detection.
# Cache keys unchanged (frozen .pkl): intent_answers / oracle_intent_answers.

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
PASSAGE_TEXT_PKL = "data/_f01_passage_text.pkl"

CELLS = {
    "AmbigQA 7B":  "data/riv_cache_test.pkl",
    "AmbigQA V3.2": "data/riv_cache_test__deepseek-ai_DeepSeek-V3.2.pkl",
    "ASQA 7B":     "data/asqa/riv_cache_test__qwen2.5-instruct.pkl",
    "ASQA V3.2":   "data/asqa/riv_cache_test__deepseek-ai_DeepSeek-V3.2.pkl",
    "SituatedQA 7B":   "data/situatedqa/riv_cache_test__qwen2.5-instruct.pkl",
    "SituatedQA V3.2": "data/situatedqa/riv_cache_test__deepseek-ai_DeepSeek-V3.2.pkl",
}


def load(p):
    o = pickle.load(open(p, "rb"))
    if isinstance(o, dict) and "records" in o:
        return list(o["records"].values())
    if isinstance(o, dict):
        return [v for k, v in o.items() if not str(k).startswith("_") and isinstance(v, dict)]
    return o


def gt(r):   return r.get("ground_truth_answers") or r.get("gt_answers") or []
def direct(r): return r.get("direct_answer") or (r.get("ans_texts") or [""])[0] or ""
def ia(r):   return r.get("intent_answers") or []


def f1(ps, g):
    ps = [str(p).strip() for p in ps if str(p).strip()]
    if not ps:
        return 0.0
    return compute_ambigqa_f1(ps[0], g) if len(ps) == 1 else compute_ambigqa_f1_multi(ps, g)


def bestsub(c, g, mx=6):
    c = _dedup_semantic([x for x in c if str(x).strip()])[:mx]
    b = 0.0
    for k in range(1, len(c) + 1):
        for idx in itertools.combinations(range(len(c)), k):
            b = max(b, f1([c[i] for i in idx], g))
    return b


class NLI:
    def __init__(self, name):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.dev).eval()
        # locate the entailment column from the model config
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.entail_idx = next(i for i, l in id2label.items() if "entail" in l)
        self.id2label = id2label

    @torch.no_grad()
    def entail_probs(self, premises, hypotheses, bs=64):
        """Return P(entailment) and argmax-is-entailment for aligned premise/hypothesis lists."""
        probs, is_top = [], []
        for s in range(0, len(premises), bs):
            enc = self.tok(premises[s:s + bs], hypotheses[s:s + bs],
                           truncation=True, max_length=256, padding=True, return_tensors="pt").to(self.dev)
            logits = self.model(**enc).logits
            p = torch.softmax(logits, -1)
            probs.extend(p[:, self.entail_idx].tolist())
            is_top.extend((p.argmax(-1) == self.entail_idx).tolist())
        return probs, is_top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=None,
                    help="If set, keep candidate when max entail-prob >= tau; else use threshold-free argmax rule.")
    a = ap.parse_args()

    ptext = pickle.load(open(PASSAGE_TEXT_PKL, "rb"))
    print(f"passage texts: {len(ptext)} | rule: {'tau=%.2f'%a.tau if a.tau is not None else 'argmax-entailment (no tuning)'}\n")
    nli = NLI(NLI_MODEL)
    print(f"NLI loaded on {nli.dev} | labels={nli.id2label} | entail_idx={nli.entail_idx}\n")

    for lab, path in CELLS.items():
        recs = load(path)
        rows = []   # per-record dict of method F1s + keep info
        for r in recs:
            q = r.get("question", "")
            g = gt(r)
            d = direct(r)
            amb = bool(r.get("is_ambiguous", False))
            cands = _dedup_semantic([d] + ia(r)) or [d]
            passages = []
            for doc in r.get("retrieved", []):
                t = ptext.get(str(doc.get("id")))
                if t:
                    title, txt = (t[1], t[0]) if isinstance(t, tuple) else ("", t)
                    passages.append((f"{title}. {txt}" if title else txt))
            passages = passages or [""]
            # build all (passage, hypothesis) pairs for this record's candidates
            prem, hyp, owner = [], [], []
            for ci, c in enumerate(cands):
                h = f"{q} The answer is {c}."
                for ps in passages:
                    prem.append(ps); hyp.append(h); owner.append(ci)
            ep, et = nli.entail_probs(prem, hyp)
            # reduce to per-candidate best passage
            best_p = [0.0] * len(cands); top_e = [False] * len(cands)
            for o, pe, te in zip(owner, ep, et):
                if pe > best_p[o]:
                    best_p[o] = pe
                top_e[o] = top_e[o] or te
            if a.tau is not None:
                keep = [c for c, bp in zip(cands, best_p) if bp >= a.tau]
            else:
                keep = [c for c, te in zip(cands, top_e) if te]
            grounded = _dedup_semantic(keep) or [d]
            rows.append(dict(
                direct=compute_ambigqa_f1(d, g),
                grounded=f1(grounded, g),
                perf_det=(f1(_dedup_semantic([d] + ia(r)) or [d], g) if amb else compute_ambigqa_f1(d, g)),
                perf_sel=bestsub([d] + ia(r), g),
                kept=len(grounded), ncand=len(cands), amb=amb,
            ))
        col = lambda k: [x[k] for x in rows]
        m = lambda k: float(np.mean(col(k)) * 100)
        amb_rows = [x for x in rows if x["amb"]]; clr_rows = [x for x in rows if not x["amb"]]
        print("=" * 78)
        print(f"[{lab}]  n={len(rows)}  (clear={len(clr_rows)}, ambiguous={len(amb_rows)})")
        print(f"  F1:  direct={m('direct'):.2f}   grounded={m('grounded'):.2f}   "
              f"perf_det={m('perf_det'):.2f}   perf_sel={m('perf_sel'):.2f}")
        c1 = bootstrap_confidence_interval(col("direct"), col("grounded"))
        c2 = bootstrap_confidence_interval(col("perf_det"), col("grounded"))
        fmt = lambda c: f"Δ={c['mean_diff']*100:+6.2f}  CI[{c['ci_lower']*100:+.2f},{c['ci_upper']*100:+.2f}]  p={c['p_value']:.3f}  {'*' if c['significant'] else 'n.s.'}"
        print(f"    grounded − direct        {fmt(c1)}")
        print(f"    grounded − perf_det  {fmt(c2)}")
        den = m("perf_sel") - m("direct")
        rec = (m("grounded") - m("direct")) / den * 100 if abs(den) > 1e-9 else float("nan")
        print(f"    realized headroom = {rec:.1f}%   (perf_sel−direct = {den:+.2f})")
        kr = np.mean([x["kept"] / max(x["ncand"], 1) for x in rows])
        kr_c = np.mean([x["kept"] / max(x["ncand"], 1) for x in clr_rows]) if clr_rows else float("nan")
        kr_a = np.mean([x["kept"] / max(x["ncand"], 1) for x in amb_rows]) if amb_rows else float("nan")
        print(f"    keep-rate: all={kr:.2f}  clear={kr_c:.2f}  ambiguous={kr_a:.2f}")


if __name__ == "__main__":
    main()
