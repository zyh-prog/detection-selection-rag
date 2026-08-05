"""
Evaluation metrics.

  1. compute_ambigqa_f1 / compute_ambigqa_f1_multi
     AmbigQA-style scoring: for each gold intent take the max token-F1 over its
     acceptable answers, then average across intents. The multi-answer variant
     uses Hungarian matching between predicted answers and gold intents, so a
     system that answers only one intent is penalised for the uncovered ones.

  2. compute_intent_coverage, semantic_dedup, retrieval metrics, and a paired
     bootstrap confidence interval / significance test.
"""

import re
import numpy as np
from collections import Counter
from typing import List, Union, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity


# ==================== dynamic retrieval threshold (variance + softmax entropy) ====================

def dynamic_threshold_from_retrieved_docs(
    retrieved_docs: List[Dict],
    base: float,
    alpha: float = 0.10,
    beta: float = 0.05,
) -> float:
    """
    Adjust the clarification-trigger threshold above `base` from the variance and
    softmax entropy of the top-k retrieval scores. Higher uncertainty -> slightly
    higher threshold -> clarification triggers more easily.
    """
    if base is None:
        base = 0.5
    base = float(base)
    alpha = float(alpha)
    beta = float(beta)

    if not retrieved_docs:
        return float(np.clip(base + 0.05, 0.22, 0.78))

    scores = np.array(
        [float(d.get("rerank_score", d.get("vector_score", 0.0))) for d in retrieved_docs],
        dtype=np.float64,
    )
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float(np.clip(base + 0.05, 0.22, 0.78))

    var = float(np.var(scores))
    var_norm = min(var / 0.25, 1.0)

    z = scores - np.max(scores)
    exp_z = np.exp(z)
    p = exp_z / (np.sum(exp_z) + 1e-12)
    ent = float(-np.sum(p * np.log(p + 1e-12)))
    h_max = float(np.log(len(scores))) if len(scores) > 1 else 1.0
    h_norm = min(max(ent / h_max, 0.0), 1.0) if h_max > 1e-9 else 0.0

    adjusted = base + alpha * var_norm + beta * h_norm
    return float(np.clip(adjusted, 0.22, 0.78))


# ==================== basics ====================

def _normalize(text: str) -> List[str]:
    """Normalize answer text (lowercase, drop articles, drop punctuation)."""
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.split()


def _token_f1(pred: str, gt: str) -> float:
    """Token-level F1 for a single string pair."""
    p_tokens = _normalize(pred)
    g_tokens = _normalize(gt)
    if not p_tokens or not g_tokens:
        return 0.0
    common = Counter(p_tokens) & Counter(g_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    prec = num_common / len(p_tokens)
    rec  = num_common / len(g_tokens)
    return 2 * prec * rec / (prec + rec)


# ==================== ground-truth structure normalization ====================

def normalize_ambigqa_ground_truth(gt: List) -> List[List[str]]:
    """
    Normalize ground_truth_answers into List[List[str]]: each element is the list
    of acceptable answer strings for one intent. Removes redundant outer nesting
    so a list is never treated as a string during F1.
    """
    if not gt:
        return []

    if isinstance(gt, list) and gt and isinstance(gt[0], str):
        return [[str(s) for s in gt if s is not None and str(s).strip()]]

    gta: List = list(gt)
    # strip single-element wrapping until string leaves or a multi-element top appears
    while (
        len(gta) == 1
        and isinstance(gta[0], list)
        and gta[0]
        and all(isinstance(x, list) for x in gta[0])
    ):
        gta = gta[0]

    out: List[List[str]] = []
    for block in gta:
        if not block:
            out.append([])
            continue
        if all(x is None or isinstance(x, str) for x in block):
            out.append([str(s) for s in block if s is not None and str(s).strip()])
        elif isinstance(block[0], list):
            for sub in block:
                if sub:
                    out.append([str(s) for s in sub if s is not None and str(s).strip()])
        else:
            out.append([str(block)])

    return out if any(out) else [[]]


# ==================== core metrics ====================

def compute_ambigqa_f1(prediction: str, gt_answers_per_intent: List[List[str]]) -> float:
    return compute_ambigqa_f1_multi([prediction], gt_answers_per_intent)


def compute_ambigqa_f1_multi(
    preds_per_intent: List[str],
    gt_answers_per_intent: List[List[str]],
    dedup_predictions: bool = True,
) -> float:
    from scipy.optimize import linear_sum_assignment

    preds = [str(p).strip() for p in preds_per_intent if str(p).strip()]
    if dedup_predictions:
        seen, unique = set(), []
        for p in preds:
            k = " ".join(p.lower().split())
            if k not in seen:
                seen.add(k)
                unique.append(p)
        preds = unique

    gt_answers_per_intent = normalize_ambigqa_ground_truth(gt_answers_per_intent)
    gt_answers_per_intent = [
        [str(a) for a in slot if a and str(a).strip()]
        for slot in gt_answers_per_intent
    ]
    gt_answers_per_intent = [slot for slot in gt_answers_per_intent if slot]

    if not preds or not gt_answers_per_intent:
        return 0.0

    n_pred = len(preds)
    n_gt = len(gt_answers_per_intent)

    score = np.zeros((n_pred, n_gt), dtype=np.float64)
    for i, pred in enumerate(preds):
        for j, ans_list in enumerate(gt_answers_per_intent):
            score[i, j] = max(_token_f1(pred, ans) for ans in ans_list)

    row_ind, col_ind = linear_sum_assignment(-score)
    matched_score = float(score[row_ind, col_ind].sum())

    return float(2.0 * matched_score / (n_pred + n_gt))

def compute_f1(
    prediction: str,
    ground_truths: Union[str, List]
) -> float:
    """
    Backward-compatible wrapper.

    If ground_truths is List[List[str]] (multi-intent structure) it routes to
    compute_ambigqa_f1; if it is List[str] (single intent, multiple aliases) it
    is wrapped as a single-intent call.
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    # multi-intent structure (List[List[str]])?
    if ground_truths and isinstance(ground_truths[0], list):
        return compute_ambigqa_f1(prediction, ground_truths)
    else:
        # single intent: wrap as [[ans1, ans2, ...]]
        flat = [str(g) for g in ground_truths if g and str(g).strip()]
        return compute_ambigqa_f1(prediction, [flat])


# ==================== intent coverage ====================

def compute_intent_coverage(
    xinference_client,
    generated_intents: List[str],
    ground_truth_intents: List[str],
    coverage_threshold: float = 0.80
) -> Tuple[float, Dict]:
    """
    Coverage of gold intents by the reverse-generated intents.

    For each gold intent, check whether any generated intent has cosine
    similarity >= threshold. Gold information is not used to pick generated
    intents, so there is no leakage.
    """
    if not generated_intents or not ground_truth_intents:
        return 0.0, {'covered': 0, 'total': len(ground_truth_intents)}

    try:
        all_texts = generated_intents + ground_truth_intents
        embeddings = xinference_client.embed(all_texts)
        gen_embs = np.array(embeddings[:len(generated_intents)])
        gt_embs  = np.array(embeddings[len(generated_intents):])
    except Exception as e:
        print(f"  [warn] intent-coverage computation failed: {e}")
        return 0.0, {'covered': 0, 'total': len(ground_truth_intents)}

    sim_matrix = cosine_similarity(gt_embs, gen_embs)

    covered_count = 0
    details = []
    for i, gt_intent in enumerate(ground_truth_intents):
        max_sim = float(np.max(sim_matrix[i]))
        best_gen_idx = int(np.argmax(sim_matrix[i]))
        is_covered = max_sim >= coverage_threshold
        if is_covered:
            covered_count += 1
        details.append({
            'gt_intent': gt_intent,
            'best_match': generated_intents[best_gen_idx],
            'similarity': max_sim,
            'is_covered': is_covered
        })

    coverage = covered_count / len(ground_truth_intents)
    return coverage, {
        'covered': covered_count,
        'total': len(ground_truth_intents),
        'coverage_rate': coverage,
        'details': details
    }


def semantic_dedup(
    xinference_client,
    intents: List[str],
    sim_threshold: float = 0.90
) -> List[str]:
    """
    Remove intents with cosine similarity >= sim_threshold, used to filter highly
    similar candidates after reverse generation.
    """
    if len(intents) <= 1:
        return intents

    try:
        embeddings = np.array(xinference_client.embed(intents))
    except Exception:
        # fallback: string dedup
        seen, unique = set(), []
        for intent in intents:
            key = intent.lower().strip()
            if key not in seen:
                unique.append(intent)
                seen.add(key)
        return unique

    sim_matrix = cosine_similarity(embeddings)
    keep = [True] * len(intents)

    for i in range(len(intents)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(intents)):
            if keep[j] and sim_matrix[i][j] >= sim_threshold:
                keep[j] = False

    return [intent for intent, k in zip(intents, keep) if k]


def compute_retrieval_metrics(
    retrieved_titles: List[str],
    ground_truth_titles: List[str]
) -> Dict:
    """Retrieval precision/recall/F1 with title normalization."""
    if not ground_truth_titles:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'hit_count': 0}

    def _norm(t):
        s = str(t).strip().strip('"').strip("'").strip()
        return s.replace('_', ' ').lower()

    retrieved_set = {_norm(t) for t in retrieved_titles}
    gt_set        = {_norm(t) for t in ground_truth_titles}
    intersection  = retrieved_set & gt_set

    precision = len(intersection) / len(retrieved_set) if retrieved_set else 0.0
    recall    = len(intersection) / len(gt_set)        if gt_set        else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1, 'hit_count': len(intersection)}

def bootstrap_confidence_interval(
    scores_a: List[float],
    scores_b: List[float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05
) -> Dict:
    """
    Paired bootstrap confidence interval + significance test for the gain of
    system b over system a.

    Returns:
        {
            'mean_diff': float,        # mean difference
            'ci_lower': float,         # (1-alpha) CI lower bound
            'ci_upper': float,         # upper bound
            'p_value': float,          # two-sided p value
            'significant': bool        # p < alpha
        }
    """
    rng = np.random.RandomState(42)
    a = np.array(scores_a)
    b = np.array(scores_b)
    observed_diff = np.mean(b) - np.mean(a)

    # paired bootstrap under H0: mean(b - a) = 0, comparing centered boot diffs to observed_diff
    diff_per_sample = b - a          # shape: (n,)
    n = len(diff_per_sample)
    centered = diff_per_sample - np.mean(diff_per_sample)  # center under H0

    boot_means = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        boot_means.append(np.mean(centered[idx]))

    boot_means = np.array(boot_means)

    # CI: percentiles of the un-centered bootstrap distribution
    raw_boot = []
    rng2 = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng2.randint(0, n, n)
        raw_boot.append(np.mean(diff_per_sample[idx]))
    raw_boot = np.array(raw_boot)

    ci_lower = float(np.percentile(raw_boot, alpha / 2 * 100))
    ci_upper = float(np.percentile(raw_boot, (1 - alpha / 2) * 100))

    # p value: fraction of |centered boot diff| >= |observed_diff| under H0
    p_value = float(np.mean(np.abs(boot_means) >= np.abs(observed_diff)))

    return {
        'mean_diff': float(observed_diff),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'significant': p_value < alpha
    }

if __name__ == "__main__":
    print("=" * 60)
    print("metrics.py quick smoke test")
    print("=" * 60)

    pred = "The first iPhone was released in 2007"
    gt_single = [["2007", "June 29, 2007"]]
    f1 = compute_ambigqa_f1(pred, gt_single)
    print(f"\n[1] single-intent F1: {f1:.3f}  (expected >0)")
    assert f1 > 0
