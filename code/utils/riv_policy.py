"""
RIV answer-collapse policies.

The main policy is intentionally conservative:
single answer by default, then append only answers that are clearly distinct
from the direct answer or from already selected clusters.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.metrics import _token_f1

_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def _norm_answer(text: str) -> str:
    return " ".join(str(text or "").lower().strip().split())


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", _norm_answer(text)))


def _years(text: str) -> set:
    return set(_YEAR_RE.findall(str(text or "")))


def _numbers(text: str) -> set:
    return set(_NUMBER_RE.findall(str(text or "")))


def unique_nonempty_answers(preds: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for pred in preds:
        key = _norm_answer(pred)
        if key and key not in seen:
            seen.add(key)
            out.append(str(pred).strip())
    return out


def hard_distinct(a: str, b: str) -> bool:
    """
    High-precision guardrail for short factual answers.

    BGE-style embedding similarity often over-merges answers such as 2007 vs 2009.
    This function forces separation when lexical evidence says the answers differ.
    """
    a = str(a or "").strip()
    b = str(b or "").strip()
    if not a or not b:
        return False
    if _norm_answer(a) == _norm_answer(b):
        return False

    ay, by = _years(a), _years(b)
    if ay and by and ay.isdisjoint(by):
        return True

    an, bn = _numbers(a), _numbers(b)
    if an and bn and an.isdisjoint(bn):
        return True

    at, bt = _tokens(a), _tokens(b)
    if len(at) <= 4 and len(bt) <= 4 and at.isdisjoint(bt):
        return True

    return False


def answer_similarity(a: str, b: str, answer_embeddings: Optional[Dict[str, List[float]]] = None) -> float:
    """
    Similarity in [0, 1]. hard_distinct overrides semantic similarity.
    """
    if not a or not b:
        return 0.0
    if _norm_answer(a) == _norm_answer(b):
        return 1.0
    if hard_distinct(a, b):
        return 0.0

    token_sim = _token_f1(a, b)
    if not answer_embeddings:
        return float(token_sim)

    ea = answer_embeddings.get(_norm_answer(a))
    eb = answer_embeddings.get(_norm_answer(b))
    if ea is None or eb is None:
        return float(token_sim)

    va = np.asarray(ea, dtype=np.float32)
    vb = np.asarray(eb, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-9:
        emb_sim = 0.0
    else:
        emb_sim = float(np.dot(va, vb) / denom)
        emb_sim = max(0.0, min(1.0, emb_sim))

    return float(max(token_sim, emb_sim))


@dataclass
class CollapsePolicy:
    tau: float = 0.80
    min_support: int = 1
    max_answers: int = 4
    require_distinct_from_direct: bool = True
    allow_singleton_hard_distinct: bool = False
    max_answer_words: int = 8
    gate_mode: str = "none"  # none, singleton, require
    gate_threshold: float = 0.50

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, obj: Dict):
        allowed = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in dict(obj).items() if k in allowed})

    @classmethod
    def from_json(cls, path: str):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data.get("policy", data))

    def save_json(self, path: str, extra: Optional[Dict] = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {"policy": self.to_dict()}
        if extra:
            payload.update(extra)
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def cluster_answers(
        answers: Sequence[str],
        answer_embeddings: Optional[Dict[str, List[float]]] = None,
        tau: float = 0.80,
) -> List[Dict]:
    clusters: List[Dict] = []
    for ans in unique_nonempty_answers(answers):
        placed = False
        for cluster in clusters:
            sim = answer_similarity(ans, cluster["representative"], answer_embeddings)
            if sim >= tau and not hard_distinct(ans, cluster["representative"]):
                cluster["members"].append(ans)
                cluster["support"] += 1
                if len(ans) < len(cluster["representative"]):
                    cluster["representative"] = ans
                placed = True
                break
        if not placed:
            clusters.append({"representative": ans, "members": [ans], "support": 1})
    return clusters


def ambiguity_gate_passed(record: Dict, threshold: float = 0.50) -> bool:
    """
    Read an ambiguity gate from cache if available.

    Supported fields:
      - ambiguity_gate: bool/int/float/str
      - gate: bool/int/float/str
      - ambiguity_gate_score: float
      - gate_score: float
      - ambiguity_gate_label: ambiguous/clear
    """
    for key in ("ambiguity_gate", "gate"):
        if key in record:
            val = record.get(key)
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                return float(val) >= threshold
            if isinstance(val, str):
                return val.strip().lower() in {"ambiguous", "amb", "true", "yes", "1"}

    for key in ("ambiguity_gate_score", "gate_score"):
        if key in record:
            try:
                return float(record.get(key)) >= threshold
            except Exception:
                pass

    label = str(record.get("ambiguity_gate_label", "")).strip().lower()
    if label:
        return label in {"ambiguous", "amb", "true", "yes", "1"}
    return False


def collapse_predictions(record: Dict, policy: CollapsePolicy) -> Tuple[List[str], Dict]:
    direct = str(record.get("direct_answer") or record.get("single_pred") or "").strip()
    intent_answers = unique_nonempty_answers(record.get("intent_answers") or record.get("multi_preds") or [])
    answer_embeddings = record.get("answer_embeddings") or {}
    gate_ok = ambiguity_gate_passed(record, threshold=float(policy.gate_threshold))

    if str(policy.gate_mode).lower() == "require" and not gate_ok:
        selected = [direct] if direct else (intent_answers[:1] or [""])
        return selected[: policy.max_answers], {
            "clusters": [],
            "n_selected": len([x for x in selected if str(x).strip()]),
            "policy": policy.to_dict(),
            "gate_passed": gate_ok,
            "gate_blocked": True,
        }

    clusters = cluster_answers(intent_answers, answer_embeddings, tau=policy.tau)
    clusters.sort(key=lambda c: (-int(c["support"]), len(c["representative"])))

    selected: List[str] = []
    if direct:
        selected.append(direct)

    for cluster in clusters:
        cand = cluster["representative"]
        if not cand:
            continue
        if policy.max_answer_words and len(str(cand).split()) > int(policy.max_answer_words):
            continue
        if len(selected) >= policy.max_answers:
            break
        if any(_norm_answer(cand) == _norm_answer(x) for x in selected):
            continue

        support_ok = int(cluster["support"]) >= int(policy.min_support)
        singleton_ok = (
                bool(policy.allow_singleton_hard_distinct)
                and int(cluster["support"]) == 1
                and bool(direct)
                and hard_distinct(cand, direct)
        )
        if str(policy.gate_mode).lower() == "singleton" and singleton_ok and not gate_ok:
            singleton_ok = False
        if not support_ok and not singleton_ok:
            continue

        if policy.require_distinct_from_direct and direct:
            sim_to_direct = answer_similarity(cand, direct, answer_embeddings)
            if sim_to_direct >= policy.tau and not hard_distinct(cand, direct):
                continue

        if any(answer_similarity(cand, prev, answer_embeddings) >= policy.tau and not hard_distinct(cand, prev)
               for prev in selected):
            continue
        selected.append(cand)

    if not selected:
        selected = intent_answers[:1] or [""]

    debug = {
        "clusters": clusters,
        "n_selected": len([x for x in selected if str(x).strip()]),
        "policy": policy.to_dict(),
        "gate_passed": gate_ok,
    }
    return selected[: policy.max_answers], debug


@dataclass
class SimilarityThresholdPolicy:
    """
    Kept only as an ablation baseline.
    """
    threshold: float = 0.66

    def predict(self, record: Dict) -> List[str]:
        eff = float((record.get("intent_features") or {}).get("eff_sim", 1.0))
        if eff < self.threshold:
            return unique_nonempty_answers(record.get("intent_answers") or []) or [record.get("direct_answer", "")]
        return [record.get("direct_answer", "")]
# ==================== backward-compatible helpers ====================

def _dedup_semantic(items, xinference_client=None, sim_threshold=0.90):
    """
    Semantic dedup helper. Falls back to string dedup when no embedding client is given.
    """
    items = [str(x).strip() for x in (items or []) if str(x).strip()]
    if len(items) <= 1:
        return items

    if xinference_client is None:
        seen, out = set(), []
        for x in items:
            k = " ".join(x.lower().split())
            if k not in seen:
                seen.add(k)
                out.append(x)
        return out

    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        embs = np.asarray(xinference_client.embed(items), dtype=np.float32)
        sim = cosine_similarity(embs)
        keep = [True] * len(items)

        for i in range(len(items)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(items)):
                if keep[j] and sim[i, j] >= sim_threshold and not hard_distinct(items[i], items[j]):
                    keep[j] = False

        return [x for x, k in zip(items, keep) if k]
    except Exception:
        seen, out = set(), []
        for x in items:
            k = " ".join(x.lower().split())
            if k not in seen:
                seen.add(k)
                out.append(x)
        return out


def f1_for(result_or_preds, gt_answers):
    """
    F1 entry point. Supports:
      - result dict: {'preds': [...]} / {'answer': '...'}
      - list[str]
      - str
    """
    from utils.metrics import compute_ambigqa_f1, compute_ambigqa_f1_multi

    if isinstance(result_or_preds, dict):
        preds = result_or_preds.get("preds")
        if preds is None:
            ans = result_or_preds.get("answer", "")
            preds = [ans] if ans else []
    elif isinstance(result_or_preds, list):
        preds = result_or_preds
    else:
        preds = [str(result_or_preds)]

    preds = [str(p).strip() for p in preds if str(p).strip()]
    if not preds:
        return 0.0
    if len(preds) == 1:
        return compute_ambigqa_f1(preds[0], gt_answers)
    return compute_ambigqa_f1_multi(preds, gt_answers)