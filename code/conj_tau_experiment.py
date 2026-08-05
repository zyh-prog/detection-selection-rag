"""conj_tau_experiment.py — does a *deployable* selector close the AmbigQA gap?

Two diagnosis-driven selectors, both computed from existing caches (self-verify
decisions are cached as `candidate_valid`; NLI entailment recomputed with the
small DeBERTa model on GPU — no FAISS/LLM re-run):

  conj  : keep direct always; add interpretation i iff  self_verify[i] AND NLI-entails[i]
  tau   : keep candidate iff NLI entail-prob >= tau   (tau tuned on dev)
  conj+tau: self_verify[i] AND NLI_bestprob[i] >= tau

We reproduce the paper's Self-verifier and NLI-argmax selectors as a correctness
check, then test whether the conjunction / calibration flips AmbigQA from the
reported negative (-0.49 / -0.57) to >= 0. set-F1, paired bootstrap (10k, seed 42).
"""
import os, sys, pickle, itertools, collections
sys.path.append(".")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.metrics import compute_ambigqa_f1, compute_ambigqa_f1_multi, bootstrap_confidence_interval
from utils.riv_policy import _dedup_semantic

GPU = "cuda:1"
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
SCORES_PKL = "data/_conj_nli_scores.pkl"
PTEXT = pickle.load(open("data/_f01_passage_text.pkl", "rb"))

TEST = {
    "AmbigQA 7B":      "data/riv_cache_test.pkl",
    "AmbigQA V3.2":    "data/riv_cache_test__deepseek-ai_DeepSeek-V3.2.pkl",
    "ASQA 7B":         "data/asqa/riv_cache_test__qwen2.5-instruct.pkl",
    "ASQA V3.2":       "data/asqa/riv_cache_test__deepseek-ai_DeepSeek-V3.2.pkl",
    "SituatedQA 7B":   "data/situatedqa/riv_cache_test__qwen2.5-instruct.pkl",
    "SituatedQA V3.2": "data/situatedqa/riv_cache_test__deepseek-ai_DeepSeek-V3.2.pkl",
}
DEV = {  # for honest tau tuning (ASQA has no dev cache)
    "AmbigQA 7B":      "data/riv_cache_dev.pkl",
    "AmbigQA V3.2":    "data/riv_cache_dev__deepseek-ai_DeepSeek-V3.2.pkl",
    "SituatedQA 7B":   "data/situatedqa/riv_cache_dev__qwen2.5-instruct.pkl",
    "SituatedQA V3.2": "data/situatedqa/riv_cache_dev__deepseek-ai_DeepSeek-V3.2.pkl",
}

def load(p):
    o = pickle.load(open(p, "rb"))
    if isinstance(o, dict) and "records" in o: return list(o["records"].values())
    if isinstance(o, dict): return [v for k, v in o.items() if not str(k).startswith("_") and isinstance(v, dict)]
    return o
def gt(r): return r.get("ground_truth_answers") or r.get("gt_answers") or []
def direct(r): return r.get("direct_answer") or (r.get("ans_texts") or [""])[0] or ""
def ia(r): return (r.get("intent_answers") or [])[:4]
def f1(ps, g):
    ps = [str(p).strip() for p in ps if str(p).strip()]
    if not ps: return 0.0
    return compute_ambigqa_f1(ps[0], g) if len(ps) == 1 else compute_ambigqa_f1_multi(ps, g)
def bestsub(c, g, mx=6):
    c = _dedup_semantic([x for x in c if str(x).strip()])[:mx]; b = 0.0
    for k in range(1, len(c) + 1):
        for idx in itertools.combinations(range(len(c)), k): b = max(b, f1([c[i] for i in idx], g))
    return b
def passages_of(r):
    ps = []
    for doc in r.get("retrieved", []):
        t = PTEXT.get(str(doc.get("id")))
        if t:
            title, txt = (t[1], t[0]) if isinstance(t, tuple) else ("", t)
            ps.append(f"{title}. {txt}" if title else txt)
    return ps or [""]

class NLI:
    def __init__(self, name, dev):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForSequenceClassification.from_pretrained(name).to(dev).eval(); self.dev = dev
        self.ei = next(i for i, l in self.m.config.id2label.items() if "entail" in l.lower())
    @torch.no_grad()
    def run(self, prem, hyp, bs=128):
        ep, top = [], []
        for i in range(0, len(prem), bs):
            enc = self.tok(prem[i:i+bs], hyp[i:i+bs], truncation=True, max_length=256,
                           padding=True, return_tensors="pt").to(self.dev)
            p = torch.softmax(self.m(**enc).logits, -1)
            ep.extend(p[:, self.ei].tolist()); top.extend((p.argmax(-1) == self.ei).tolist())
        return ep, top

def build_scores(cells, nli):
    """Per record: d, ia(<=4), cv(4 bool), bestp/top for items=[d]+ia, plus
    f_direct / f_perfect (Perfect detection) / f_oracle (Perfect selection) F1, amb.
    NOTE: these row keys are frozen in SCORES_PKL (_conj_nli_scores.pkl) — do not rename."""
    out = {}
    for lab, path in cells.items():
        if not os.path.exists(path): continue
        recs = load(path)
        prem, hyp, meta, items_all = [], [], [], []
        for ri, r in enumerate(recs):
            items = [direct(r)] + list(ia(r))
            items_all.append(items)
            q = r.get("question", "")
            ps = passages_of(r)
            for ii, it in enumerate(items):
                h = f"{q} The answer is {it}."
                for pp in ps:
                    prem.append(pp); hyp.append(h); meta.append((ri, ii))
        ep, top = nli.run(prem, hyp)
        bestp = collections.defaultdict(float); topany = collections.defaultdict(bool)
        for (ri, ii), e, t in zip(meta, ep, top):
            if e > bestp[(ri, ii)]: bestp[(ri, ii)] = e
            topany[(ri, ii)] = topany[(ri, ii)] or t
        rows = []
        for ri, r in enumerate(recs):
            items = items_all[ri]; n = len(items); g = gt(r); d = items[0]; iaa = items[1:]
            cv = (r.get("candidate_valid") or [False]*len(iaa))[:len(iaa)]
            amb = bool(r.get("is_ambiguous", False))
            rows.append(dict(
                d=d, ia=iaa, cv=[bool(x) for x in cv], amb=amb, g=g,
                bestp=[bestp[(ri, ii)] for ii in range(n)],
                top=[bool(topany[(ri, ii)]) for ii in range(n)],
                f_direct=compute_ambigqa_f1(d, g),
                f_perfect=(f1(_dedup_semantic([d]+iaa) or [d], g) if amb else compute_ambigqa_f1(d, g)),
                f_oracle=bestsub([d]+iaa, g)))
        out[lab] = rows
        print(f"  scored {lab}: n={len(rows)}", flush=True)
    return out

def pred(rec, rule, tau=0.5):
    d, iaa, cv, bestp, top = rec["d"], rec["ia"], rec["cv"], rec["bestp"], rec["top"]
    items = [d] + iaa; N = len(iaa)
    if rule == "direct":  return [d]
    if rule == "selfver": return _dedup_semantic([d] + [iaa[i] for i in range(N) if cv[i]]) or [d]
    if rule == "nli_arg": return _dedup_semantic([items[j] for j in range(len(items)) if top[j]]) or [d]
    if rule == "conj_arg":return _dedup_semantic([d] + [iaa[i] for i in range(N) if cv[i] and top[i+1]]) or [d]
    if rule == "nli_tau": return _dedup_semantic([items[j] for j in range(len(items)) if bestp[j] >= tau]) or [d]
    if rule == "conj_tau":return _dedup_semantic([d] + [iaa[i] for i in range(N) if cv[i] and bestp[i+1] >= tau]) or [d]

def meanf1(rows, rule, tau=0.5):
    return float(np.mean([f1(pred(r, rule, tau), r["g"]) for r in rows]) * 100)

def tune_tau(dev_rows_all, rule, grid):
    best, bt = -1, grid[0]
    for t in grid:
        v = np.mean([f1(pred(r, rule, t), r["g"]) for rows in dev_rows_all for r in rows])
        if v > best: best, bt = v, t
    return bt, best*100

def evalcell(rows, tau_nli, tau_conj):
    direct_f = [r["f_direct"] for r in rows]
    mdirect = float(np.mean(direct_f)*100)
    moracle = float(np.mean([r["f_oracle"] for r in rows])*100)
    out = {"Direct": (mdirect, None)}
    for name, rule, tau in [("SelfVer","selfver",0), ("NLI-arg","nli_arg",0),
                            ("Conj-arg","conj_arg",0), ("NLI-tau","nli_tau",tau_nli),
                            ("Conj-tau","conj_tau",tau_conj)]:
        fl = [f1(pred(r, rule, tau), r["g"]) for r in rows]
        mm = float(np.mean(fl)*100)
        ci = bootstrap_confidence_interval(direct_f, fl)
        realized = (mm-mdirect)/(moracle-mdirect)*100 if moracle-mdirect > 1e-9 else float("nan")
        out[name] = (mm, ci, realized)
    # multi-rate (|pred|>1) clear vs amb for conj_arg
    clr = [r for r in rows if not r["amb"]]; amb = [r for r in rows if r["amb"]]
    mr = lambda rs, rule, tau: (np.mean([len(set(pred(r,rule,tau)))>1 for r in rs]) if rs else float("nan"))
    out["_multi"] = dict(
        conj_clear=mr(clr,"conj_arg",0), conj_amb=mr(amb,"conj_arg",0),
        self_clear=mr(clr,"selfver",0), self_amb=mr(amb,"selfver",0))
    out["_oracle"] = moracle
    return out

def main():
    if os.path.exists(SCORES_PKL):
        print("loading cached NLI scores", flush=True)
        S = pickle.load(open(SCORES_PKL, "rb"))
    else:
        print(f"computing NLI on {GPU} ...", flush=True)
        nli = NLI(NLI_MODEL, GPU)
        S = build_scores({**TEST, **{"DEV::"+k: v for k, v in DEV.items()}}, nli)
        pickle.dump(S, open(SCORES_PKL, "wb"))
    dev_all = [S["DEV::"+k] for k in DEV if "DEV::"+k in S]
    grid = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]
    tau_nli, vn = tune_tau(dev_all, "nli_tau", grid)
    tau_conj, vc = tune_tau(dev_all, "conj_tau", grid)
    print(f"\nGlobal tau tuned on dev ({len(dev_all)} dev cells): "
          f"NLI-tau*={tau_nli} (dev F1 {vn:.2f}) | Conj-tau*={tau_conj} (dev F1 {vc:.2f})\n")
    print("="*100)
    hdr = f"{'cell':14s} {'Direct':>7s} {'SelfVer':>16s} {'NLI-arg':>16s} {'Conj-arg':>16s} {'NLI-t':>16s} {'Conj-t':>16s}"
    for lab in TEST:
        if lab not in S: continue
        e = evalcell(S[lab], tau_nli, tau_conj)
        print(f"\n### {lab}   (oracle={e['_oracle']:.2f}, direct={e['Direct'][0]:.2f})")
        for name in ["SelfVer","NLI-arg","Conj-arg","NLI-tau","Conj-tau"]:
            mm, ci, rl = e[name]
            d = ci["mean_diff"]*100; lo=ci["ci_lower"]*100; hi=ci["ci_upper"]*100
            sig = "*" if ci["significant"] else " "
            print(f"   {name:9s} F1={mm:5.2f}  Δ={d:+5.2f}{sig} CI[{lo:+5.2f},{hi:+5.2f}] p={ci['p_value']:.3f}  realized={rl:+5.1f}%")
        m = e["_multi"]
        print(f"   multi-rate(|pred|>1)  conj: clear={m['conj_clear']:.2f} amb={m['conj_amb']:.2f}  |  self: clear={m['self_clear']:.2f} amb={m['self_amb']:.2f}")
    print("\n(Validation: SelfVer/NLI-arg should match paper Tables 1/4: AmbigQA7B ≈ -0.49 / -0.57.)")

if __name__ == "__main__":
    main()
