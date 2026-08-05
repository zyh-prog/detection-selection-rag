import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append(".")

from config import config
from utils.xinference_client import XInferenceClient
from utils.metrics import compute_ambigqa_f1, compute_ambigqa_f1_multi
from utils.riv_policy import hard_distinct, _dedup_semantic


PROMPT = """Original question:
{question}

Candidate interpretation:
{intent}

Candidate answer:
{answer}

Is the candidate interpretation a valid disambiguation of the original question?

Reply VALID only if:
- it preserves the original asked relation/property;
- it only resolves an ambiguous entity/person/work/time/version;
- it could be one plausible meaning of the original question;
- the answer is factual and not a refusal.

Reply INVALID if:
- it changes the topic;
- it asks about a different show/person/event;
- it is merely related retrieval noise;
- the answer is unknown/not specified/refusal.

Reply one token: VALID or INVALID.
"""


def load_records(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "records" in obj:
        return obj, obj["records"]
    if isinstance(obj, dict):
        return obj, obj
    raise TypeError(type(obj))


def save_cache(obj, path):
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    Path(tmp).replace(path)


def gt(r):
    return r.get("ground_truth_answers") or r.get("gt_answers") or []


def direct(r):
    return r.get("direct_answer") or r.get("ans_texts", [""])[0] or ""


def intents(r):
    return r.get("generated_intents") or r.get("interpretations") or []


def answers(r):
    return r.get("intent_answers") or []


def f1(preds, g):
    preds = [str(p).strip() for p in preds if str(p).strip()]
    if not preds:
        return 0.0
    if len(preds) == 1:
        return compute_ambigqa_f1(preds[0], g)
    return compute_ambigqa_f1_multi(preds, g)


def valid_answer(a):
    s = str(a or "").lower()
    bad = ["not specified", "unknown", "unable", "cannot", "no answer", "not provided"]
    return bool(s.strip()) and not any(x in s for x in bad)


def verify_one(xi, q, intent, answer):
    if not valid_answer(answer):
        return False, "INVALID_REFUSAL"
    try:
        resp = xi.chat(
            [{"role": "user", "content": PROMPT.format(question=q, intent=intent, answer=answer)}],
            model=config.LLM_MODEL,
            temperature=0.0,
            max_tokens=4,
        )
        raw = (resp or "").strip()
        return raw.upper().startswith("VALID"), raw
    except Exception as e:
        return False, f"ERROR: {e}"


def preds_for_policy(r, mode):
    d = direct(r)
    vals = r.get("candidate_valid", [])
    cand = []

    for ok, a in zip(vals, answers(r)):
        if ok and valid_answer(a):
            cand.append(a)

    cand = _dedup_semantic(cand)

    if mode == "direct":
        return [d]

    if mode == "verified_only":
        return cand or [d]

    if mode == "replace_if_multi":
        return cand if len(cand) >= 2 else [d]

    if mode == "direct_plus_hard":
        out = [d]
        for a in cand:
            if hard_distinct(a, d) and a not in out:
                out.append(a)
        return out

    if mode == "direct_plus_verified":
        out = [d]
        for a in cand:
            if a not in out:
                out.append(a)
        return _dedup_semantic(out)

    return [d]


def evaluate(records):
    modes = [
        "direct",
        "verified_only",
        "replace_if_multi",
        "direct_plus_hard",
        "direct_plus_verified",
    ]

    print("\n" + "=" * 72)
    print("Verified candidate policy evaluation")
    print("=" * 72)
    print(f"{'mode':<24} {'F1':>8} {'multi':>8} {'amb_rec':>8} {'false':>8} {'avg_n':>8}")

    for mode in modes:
        fs, ns = [], []
        multi = false = amb_multi = amb_total = 0

        for r in records:
            p = preds_for_policy(r, mode)
            is_multi = len(p) > 1
            amb = bool(r.get("is_ambiguous", False))

            fs.append(f1(p, gt(r)))
            ns.append(len(p))

            if amb:
                amb_total += 1
            if is_multi:
                multi += 1
                if amb:
                    amb_multi += 1
                else:
                    false += 1

        n = max(len(records), 1)
        clear_total = max(n - amb_total, 1)
        print(
            f"{mode:<24} {np.mean(fs)*100:>8.2f} "
            f"{multi/n*100:>8.2f} "
            f"{amb_multi/max(amb_total,1)*100:>8.2f} "
            f"{false/clear_total*100:>8.2f} "
            f"{np.mean(ns):>8.2f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/riv_cache_dev.pkl")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    cache_obj, rec_map = load_records(args.cache)
    from utils.deepseek_client import DeepSeekHybridClient
    xi = (DeepSeekHybridClient(config.DEEPSEEK_BASE_URL, config.DEEPSEEK_API_KEY, config.XINFERENCE_BASE_URL)
        if str(getattr(config,"LLM_BACKEND","xinference"))=="deepseek"
        else XInferenceClient(config.XINFERENCE_BASE_URL))

    for _, r in tqdm(list(rec_map.items()), desc="verify candidates"):
        if not args.refresh and "candidate_valid" in r:
            continue

        valids, raws = [], []
        for it, ans in zip(intents(r), answers(r)):
            ok, raw = verify_one(xi, r.get("question", ""), it, ans)
            valids.append(bool(ok))
            raws.append(raw)

        r["candidate_valid"] = valids
        r["candidate_valid_raw"] = raws

    save_cache(cache_obj, args.cache)
    evaluate(list(rec_map.values()))


if __name__ == "__main__":
    main()