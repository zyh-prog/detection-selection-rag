import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

sys.path.append(".")

from config import config
from utils.data_loader import load_json, save_json
from utils.metrics import (
    bootstrap_confidence_interval,
    compute_ambigqa_f1,
    compute_ambigqa_f1_multi,
)
from utils.riv_policy import CollapsePolicy, SimilarityThresholdPolicy, collapse_predictions

import re

def model_slug() -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(getattr(config, "LLM_MODEL", "model")))

def cache_paths(cache_dir):
    slug = model_slug()
    base = Path(cache_dir)
    return str(base / f"riv_cache_dev__{slug}.pkl"), str(base / f"riv_cache_test__{slug}.pkl")

def build_client():
    if str(getattr(config, "LLM_BACKEND", "xinference")) == "deepseek":
        from utils.deepseek_client import DeepSeekHybridClient
        return DeepSeekHybridClient(config.DEEPSEEK_BASE_URL, config.DEEPSEEK_API_KEY,
                                    config.XINFERENCE_BASE_URL)
    try:
        from utils.hybrid_client import HybridClient
        return HybridClient(config.XINFERENCE_BASE_URL,
                            device=getattr(config, "HYBRID_EMBED_DEVICE", "cuda:1"))
    except Exception:
        from utils.xinference_client import XInferenceClient
        return XInferenceClient(config.XINFERENCE_BASE_URL)

def load_pickle(path: str, default=None):
    p = Path(path)
    if not p.exists():
        return default if default is not None else {}
    with open(p, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.replace(p)


def make_split(samples: Sequence[Dict], dev_size: int, seed: int = 42):
    ids = [str(s["id"]) for s in samples]
    rng = random.Random(seed)
    rng.shuffle(ids)
    return {"dev": ids[:dev_size], "test": ids[dev_size:]}



def normalize_cache_to_records(cache_obj) -> Dict[str, Dict]:
    """
    Accept three cache layouts:
    1. {"records": {id: record}}
    2. {id: record}
    3. [record, record, ...]
    """
    if isinstance(cache_obj, dict) and "records" in cache_obj:
        raw = cache_obj.get("records", {})
        return {str(k): v for k, v in raw.items() if isinstance(v, dict)}

    if isinstance(cache_obj, dict):
        return {
            str(k): v
            for k, v in cache_obj.items()
            if not str(k).startswith("_") and isinstance(v, dict)
        }

    if isinstance(cache_obj, list):
        out = {}
        for i, v in enumerate(cache_obj):
            if isinstance(v, dict):
                out[str(v.get("id", i))] = v
        return out

    return {}


def load_records_for_split(cache_path: str, ids: Iterable[str]) -> List[Dict]:
    cache = load_pickle(cache_path, default={})
    records = normalize_cache_to_records(cache)

    wanted = [str(i) for i in ids]
    present = [i for i in wanted if i in records]
    selected = [records[i] for i in present]

    if records:
        coverage = len(present) / max(len(wanted), 1)
        if coverage < 1.0:
            print(f"  [warn] {Path(cache_path).name}: split hit {len(present)}/{len(wanted)} "
                  f"({coverage*100:.0f}%)")
        if coverage < 0.5:
            raise RuntimeError(
                f"{cache_path}: only {len(present)}/{len(wanted)} split ids hit; "
                f"annotated/split/cache are likely from different sources. Refusing to "
                f"silently fall back to the full set (would contaminate dev/test). "
                f"Check the data sources or re-collect with the matching split."
            )
    return selected


def record_id(r: Dict) -> str:
    return str(r.get("id", r.get("sample_id", "")))


def direct_answer(r: Dict) -> str:
    return (
        r.get("direct_answer")
        or r.get("single_pred")
        or r.get("vanilla_answer")
        or r.get("answer")
        or ""
    )


def intent_answers(r: Dict) -> List[str]:
    return (
        r.get("intent_answers")
        or r.get("multi_preds")
        or r.get("multi_answers")
        or r.get("preds")
        or []
    )


# Accessor for the per-gold-intent answers (paper term: "Gold interpretation").
# Name + cache keys are kept as-is: they are frozen in the persisted .pkl caches.
def oracle_answers(r: Dict) -> List[str]:
    return (
        r.get("oracle_intent_answers")
        or r.get("oracle_answers")
        or r.get("gt_intent_answers_pred")
        or []
    )


def gt_answers(r: Dict):
    return r.get("ground_truth_answers") or r.get("gt_answers") or []


def is_ambiguous(r: Dict) -> bool:
    return bool(r.get("is_ambiguous", False))


def ambiguity_level(r: Dict) -> int:
    if "ambiguity_level" in r:
        return int(r.get("ambiguity_level") or 1)
    gt = gt_answers(r)
    return len(gt) if isinstance(gt, list) and gt else 1


def f1_multi(preds: Sequence[str], gt) -> float:
    preds = [str(p).strip() for p in preds if str(p).strip()]
    if not preds:
        return 0.0
    if len(preds) == 1:
        return compute_ambigqa_f1(preds[0], gt)
    return compute_ambigqa_f1_multi(preds, gt)


def score_direct(r: Dict) -> float:
    return compute_ambigqa_f1(direct_answer(r), gt_answers(r))


def score_gold_interp(r: Dict) -> float:
    preds = oracle_answers(r)
    if not preds:
        preds = intent_answers(r)
    preds = [str(p).strip() for p in preds if str(p).strip()]
    return f1_multi(preds, gt_answers(r))


def score_collapse(r: Dict, policy: CollapsePolicy) -> float:
    preds, _ = collapse_predictions(r, policy)
    return f1_multi(preds, gt_answers(r))


def score_similarity(r: Dict, threshold: float = 0.66) -> float:
    policy = SimilarityThresholdPolicy(threshold=threshold)
    return f1_multi(policy.predict(r), gt_answers(r))


def summarize_scores(records: Sequence[Dict], scorer) -> Dict:
    vals = [float(scorer(r)) for r in records]
    return {
        "f1": float(np.mean(vals) * 100) if vals else 0.0,
        "f1_scores_raw": vals,
        "n_samples": len(records),
    }


def print_oracle_gate(records: Sequence[Dict], label: str) -> Dict:
    vanilla = summarize_scores(records, score_direct)
    gold_interp = summarize_scores(records, score_gold_interp)
    gain = gold_interp["f1"] - vanilla["f1"]

    print("\n" + "=" * 64)
    print(f"[realizable upper bound — {label} (n={len(records)})]  go/no-go gate")
    print("=" * 64)
    print(f"  Vanilla (single answer)    F1 = {vanilla['f1']:.1f}%")
    print(f"  Gold-Interp (gold intents) F1 = {gold_interp['f1']:.1f}%")
    print(f"  -> realizable gain (gold_interp - vanilla) = {gain:+.1f}%")

    if gain < 3:
        print("  [warn] go/no-go: <3%; stop tuning policy, focus on retrieval / upper-bound analysis.")
    elif gain < 8:
        print("  [warn] go/no-go: 3%-8%; lightweight work is ok, but emphasize the retrieval bottleneck.")
    else:
        print("  [ok] go/no-go: substantial headroom, worth continuing policy calibration.")

    return {
        "vanilla": vanilla,
        "gold_interp": gold_interp,
        "realizable_gain": gain,
    }


def eval_policy(records: Sequence[Dict], policy: CollapsePolicy) -> Dict:
    f1s = []
    pred_counts = []
    multi_count = 0
    false_alarm = 0
    amb_total = 0
    amb_multi = 0
    details = []

    for r in records:
        preds, debug = collapse_predictions(r, policy)
        preds = [str(p).strip() for p in preds if str(p).strip()]
        if not preds:
            preds = [direct_answer(r)]

        f1 = f1_multi(preds, gt_answers(r))
        is_multi = len(preds) > 1
        amb = is_ambiguous(r)

        f1s.append(f1)
        pred_counts.append(len(preds))

        if amb:
            amb_total += 1
        if is_multi:
            multi_count += 1
            if amb:
                amb_multi += 1
            else:
                false_alarm += 1

        rid = record_id(r)
        details.append(
            {
                "id": rid,
                "query": r.get("question", r.get("query", "")),
                "is_ambiguous": amb,
                "ambiguity_level": ambiguity_level(r),
                "status": "multi_answer" if is_multi else "single_answer",
                "f1": f1,
                "prediction": (
                    "\n".join(f"{i + 1}. {p}" for i, p in enumerate(preds))
                    if is_multi
                    else preds[0]
                ),
                "preds": preds,
                "direct_answer": direct_answer(r),
                "intent_answers": intent_answers(r),
                "generated_intents": r.get("generated_intents", r.get("candidate_intents", [])),
                "collapse_debug": debug,
            }
        )

    n = max(len(records), 1)
    return {
        "avg_f1": float(np.mean(f1s) * 100) if f1s else 0.0,
        "f1_scores_raw": f1s,
        "multi_answer_rate": multi_count / n * 100,
        "false_alarm_rate": false_alarm / n * 100,
        "ambiguity_recall": amb_multi / max(amb_total, 1) * 100,
        "avg_prediction_count": float(np.mean(pred_counts)) if pred_counts else 0.0,
        "details": details,
    }


def calibrate_policy(records: Sequence[Dict]) -> Dict:
    candidates = []

    has_gate = any(
        any(k in r for k in (
            "ambiguity_gate",
            "gate",
            "ambiguity_gate_score",
            "gate_score",
            "ambiguity_gate_label",
        ))
        for r in records
    )
    gate_modes = ["none", "singleton", "require"] if has_gate else ["none"]

    for tau in np.arange(0.55, 0.951, 0.05):
        for support in [1, 2]:
            for max_answers in [2, 3, 4]:
                for singleton_hard in [False, True]:
                    for max_words in [6, 8, 12]:
                        for gate_mode in gate_modes:
                            policy = CollapsePolicy(
                                tau=float(round(tau, 3)),
                                min_support=int(support),
                                max_answers=int(max_answers),
                                require_distinct_from_direct=True,
                                allow_singleton_hard_distinct=bool(singleton_hard),
                                max_answer_words=int(max_words),
                                gate_mode=gate_mode,
                            )
                            metrics = eval_policy(records, policy)

                            # F1 is the main objective; false alarm is a constraint
                            # (so a 0-false no-op policy cannot always win).
                            f1 = metrics["avg_f1"]
                            false = metrics["false_alarm_rate"]
                            amb_rec = metrics["ambiguity_recall"]
                            multi = metrics["multi_answer_rate"]

                            # soft constraint: prefer false <= 20; penalize beyond that.
                            false_penalty = max(0.0, false - 20.0) * 0.20

                            # mild encouragement to actually trigger ambiguity (avoid multi~0 policies).
                            recall_bonus = min(amb_rec, 50.0) * 0.01

                            objective = f1 - false_penalty + recall_bonus

                            candidates.append(
                                (
                                    objective,
                                    f1,
                                    amb_rec,
                                    -false,
                                    multi,
                                    policy,
                                    metrics,
                                )
                            )

    # sort by objective, then F1, then amb_rec, then low false.
    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    best = candidates[0]

    # also keep the pure-F1 best, to see whether the constraint is holding F1 down.
    by_f1 = sorted(candidates, key=lambda x: (x[1], x[2], x[3]), reverse=True)[0]

    return {
        "policy": best[5],
        "dev_metrics": best[6],
        "objective": best[0],
        "best_by_f1": {
            "objective": float(by_f1[0]),
            "f1": float(by_f1[1]),
            "ambiguity_recall": float(by_f1[2]),
            "false_alarm_rate": float(-by_f1[3]),
            "multi_answer_rate": float(by_f1[4]),
            "policy": by_f1[5].to_dict(),
        },
        "top10": [
            {
                "objective": float(x[0]),
                "f1": float(x[1]),
                "ambiguity_recall": float(x[2]),
                "false_alarm_rate": float(-x[3]),
                "multi_answer_rate": float(x[4]),
                "policy": x[5].to_dict(),
                "metrics": {
                    "avg_f1": x[6]["avg_f1"],
                    "multi_answer_rate": x[6]["multi_answer_rate"],
                    "false_alarm_rate": x[6]["false_alarm_rate"],
                    "ambiguity_recall": x[6]["ambiguity_recall"],
                    "avg_prediction_count": x[6]["avg_prediction_count"],
                },
            }
            for x in candidates[:10]
        ],
    }


def vanilla_details(records: Sequence[Dict]) -> List[Dict]:
    out = []
    for r in records:
        out.append(
            {
                "id": record_id(r),
                "query": r.get("question", r.get("query", "")),
                "is_ambiguous": is_ambiguous(r),
                "ambiguity_level": ambiguity_level(r),
                "status": "single_answer",
                "f1": score_direct(r),
                "prediction": direct_answer(r),
            }
        )
    return out


def by_ambiguity_level(records: Sequence[Dict], van_details: List[Dict], riv_details: List[Dict]) -> Dict:
    v = {str(d["id"]): d["f1"] for d in van_details}
    r = {str(d["id"]): d["f1"] for d in riv_details}

    groups = {
        "clear": [x for x in records if ambiguity_level(x) == 1],
        "light": [x for x in records if ambiguity_level(x) == 2],
        "heavy": [x for x in records if ambiguity_level(x) >= 3],
    }

    out = {}
    for name, xs in groups.items():
        ids = [record_id(x) for x in xs]
        vf = [v[i] for i in ids if i in v]
        rf = [r[i] for i in ids if i in r]
        out[name] = {
            "vanilla": float(np.mean(vf) * 100) if vf else 0.0,
            "riv": float(np.mean(rf) * 100) if rf else 0.0,
            "improvement": float((np.mean(rf) - np.mean(vf)) * 100) if vf and rf else 0.0,
            "n_samples": len(xs),
        }
    return out


def ablation_table(records: Sequence[Dict], best_policy: CollapsePolicy) -> Dict:
    variants = {
        "Complete RIV-Agent": lambda r: score_collapse(r, best_policy),
        "Never multi-answer": score_direct,
        "Always multi-answer": lambda r: f1_multi(intent_answers(r), gt_answers(r)),
        "Gold-Interp upper": score_gold_interp,
        "Sim-threshold 0.66 baseline": lambda r: score_similarity(r, 0.66),
    }

    results = {}
    for name, fn in variants.items():
        vals = [float(fn(r)) for r in records]
        results[name] = {
            "f1": float(np.mean(vals) * 100) if vals else 0.0,
            "n_samples": len(records),
        }

    base = results["Complete RIV-Agent"]["f1"]
    for name, m in results.items():
        if name != "Complete RIV-Agent":
            m["f1_drop"] = base - m["f1"]

    return results


def maybe_collect(split: Dict, samples: List[Dict], args):
    if args.collect == "none":
        return

    from implement_systems import EnhancedRetriever
    from utils.riv_features import RIVFeatureExtractor, collect_cache

    client = build_client()
    retriever = EnhancedRetriever(client)
    extractor = RIVFeatureExtractor(retriever, client, use_rerank=True)

    cache_dir = Path(args.cache_dir)
    dev_cache, test_cache = cache_paths(args.cache_dir)

    if args.collect in ("dev", "all"):
        collect_cache(extractor, samples, split["dev"], dev_cache, refresh=args.refresh)
    if args.collect in ("test", "all"):
        collect_cache(extractor, samples, split["test"], test_cache, refresh=args.refresh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", choices=["none", "dev", "test", "all"], default="dev")
    parser.add_argument("--eval-split", choices=["dev", "test"], default="dev")
    parser.add_argument("--dev-size", type=int, default=100)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--split-path", default=f"{config.DATA_DIR}/riv_eval_split.json")
    parser.add_argument("--cache-dir", default=config.DATA_DIR)
    parser.add_argument("--policy-out", default=f"{config.DATA_DIR}/riv_collapse_policy.json")
    parser.add_argument("--out", default="")
    parser.add_argument("--data-path", default=config.ANNOTATED_DATA_PATH)
    args = parser.parse_args()

    samples = load_json(args.data_path)
    if int(getattr(config, "EVAL_SAMPLE_SIZE", len(samples))) < len(samples):
        samples = samples[: int(config.EVAL_SAMPLE_SIZE)]

    split_path = Path(args.split_path)
    if split_path.exists():
        split = load_json(str(split_path))
    else:
        split = make_split(
            samples,
            dev_size=args.dev_size,
            seed=int(getattr(config, "RANDOM_SEED", 42)),
        )
        save_json(split, str(split_path))

    maybe_collect(split, samples, args)

    cache_dir = Path(args.cache_dir)
    dev_cache, test_cache = cache_paths(args.cache_dir)

    dev_records = load_records_for_split(dev_cache, split["dev"])
    if not dev_records:
        raise RuntimeError(
            "Dev cache is empty. Run: python run_riv_pipeline.py --collect dev --eval-split dev"
        )

    dev_gate = print_oracle_gate(dev_records, "dev")
    calibration = calibrate_policy(dev_records)
    best_policy: CollapsePolicy = calibration["policy"]

    best_policy.save_json(
        args.policy_out,
        extra={
            "dev_gate": dev_gate,
            "dev_metrics": {
                "avg_f1": calibration["dev_metrics"]["avg_f1"],
                "multi_answer_rate": calibration["dev_metrics"]["multi_answer_rate"],
                "false_alarm_rate": calibration["dev_metrics"]["false_alarm_rate"],
                "ambiguity_recall": calibration["dev_metrics"]["ambiguity_recall"],
                "avg_prediction_count": calibration["dev_metrics"]["avg_prediction_count"],
            },
            "top10": calibration["top10"],
        },
    )

    print("\n[Dev calibration top-10]")
    print(f"{'rank':>4} {'F1':>8} {'multi':>8} {'amb_rec':>8} {'false':>8} {'avg_n':>7} policy")
    for i, row in enumerate(calibration["top10"], 1):
        m = row["metrics"]
        print(
            f"{i:>4} {m['avg_f1']:>8.2f} {m['multi_answer_rate']:>8.2f} "
            f"{m['ambiguity_recall']:>8.2f} {m['false_alarm_rate']:>8.2f} "
            f"{m['avg_prediction_count']:>7.2f} {row['policy']}"
        )

    eval_cache = dev_cache if args.eval_split == "dev" else test_cache
    eval_ids = split[args.eval_split]
    records = load_records_for_split(eval_cache, eval_ids)
    if not records:
        raise RuntimeError(
            f"{args.eval_split} cache is empty. "
            f"Run: python run_riv_pipeline.py --collect {args.eval_split} --eval-split {args.eval_split}"
        )

    print_oracle_gate(records, args.eval_split)

    riv = eval_policy(records, best_policy)
    van = vanilla_details(records)

    van_scores = [d["f1"] for d in van]
    riv_scores = riv["f1_scores_raw"]

    sig = bootstrap_confidence_interval(van_scores, riv_scores)

    vanilla_f1 = float(np.mean(van_scores) * 100) if van_scores else 0.0
    final = {
        "split": args.eval_split,
        "policy": best_policy.to_dict(),
        "overall": {
            "vanilla_f1": vanilla_f1,
            "riv_f1": riv["avg_f1"],
            "f1_improvement": riv["avg_f1"] - vanilla_f1,
            "clarification_rate": riv["multi_answer_rate"],
            "false_alarm_rate": riv["false_alarm_rate"],
            "avg_prediction_count": riv["avg_prediction_count"],
        },
        "intent_matching": {
            "ambiguity_recall": riv["ambiguity_recall"],
            "clarification_precision": (
                (riv["multi_answer_rate"] - riv["false_alarm_rate"])
                / max(riv["multi_answer_rate"], 1e-9)
                * 100
            ),
        },
        "by_ambiguity_level": by_ambiguity_level(records, van, riv["details"]),
        "significance": sig,
        "ablation": ablation_table(records, best_policy),
        "detailed_results": {
            "vanilla_f1_details": van,
            "riv_f1_details": riv["details"],
        },
    }

    out = args.out or f"{config.DATA_DIR}/evaluation_results_v3_{args.eval_split}.json"
    save_json(final, out)
    save_json(final["ablation"], f"{config.DATA_DIR}/ablation_results_v3_{args.eval_split}.json")

    print("\n[final results]")
    print(f"  split:      {args.eval_split}, n={len(records)}")
    print(f"  Vanilla F1: {final['overall']['vanilla_f1']:.2f}%")
    print(f"  RIV F1:     {final['overall']['riv_f1']:.2f}%")
    print(f"  Δ:          {final['overall']['f1_improvement']:+.2f}%")
    print(f"  multi rate: {final['overall']['clarification_rate']:.2f}%")
    print(f"  false rate: {final['overall']['false_alarm_rate']:.2f}%")
    print(f"  amb recall: {final['intent_matching']['ambiguity_recall']:.2f}%")
    print(f"  p-value:    {sig['p_value']:.4f}")
    print(f"  saved:      {out}")


if __name__ == "__main__":
    main()