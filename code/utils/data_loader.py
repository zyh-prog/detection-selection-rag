"""
Data loading and processing.
"""

import json
from typing import List, Dict
from pathlib import Path

from utils.metrics import normalize_ambigqa_ground_truth


def _normalize_qa_answer_field(blob) -> List[str]:
    """Normalize one AmbigQA `answer` field into a flat list of answer aliases."""
    if not blob or not isinstance(blob, list):
        return []
    if isinstance(blob[0], str):
        return [str(s) for s in blob if s is not None and str(s).strip()]
    groups = normalize_ambigqa_ground_truth(blob)
    if not groups:
        return []
    if len(groups) == 1:
        return groups[0]
    return [x for g in groups for x in g]


def _ans_key(ans_list):
    """Normalized fingerprint of an answer set, used to deduplicate intents."""
    return frozenset(a.lower().strip() for a in ans_list if a and a.strip())


def _dedup_intents(intents, answers):
    """Merge intents whose answer sets are effectively identical (filters spurious ambiguity)."""
    seen, out_i, out_a = {}, [], []
    for q, a in zip(intents, answers):
        key = _ans_key(a)
        if not key:
            continue
        if key in seen:
            idx = seen[key]
            for x in a:
                if x not in out_a[idx]:
                    out_a[idx].append(x)
        else:
            seen[key] = len(out_a)
            out_i.append(q)
            out_a.append(list(a))
    return out_i, out_a


def load_ambigqa_dataset(dataset, split: str = 'validation'):
    """
    Parse AmbigQA: each disambiguation is one independent intent with its own
    answers. is_ambiguous is set when there are >= 2 intents; ambiguity_level is
    the number of intents (the most-disambiguated round is taken as canonical).
    """
    processed = []
    for item in dataset[split]:
        ann = item['annotations']
        types = ann.get('type', []) or []
        qa_pairs_per_round = ann.get('qaPairs', []) or []
        answer_per_round = ann.get('answer', []) or []

        # parse disambiguations across all multipleQAs rounds; keep the round with the most intents
        best_intents, best_answers = [], []
        for k, t in enumerate(types):
            if t != 'multipleQAs' or k >= len(qa_pairs_per_round):
                continue
            qp = qa_pairs_per_round[k] or {}
            qs = qp.get('question') or []
            ans = qp.get('answer') or []
            cur_i, cur_a = [], []
            for i in range(min(len(qs), len(ans))):     # question[i] <-> answer[i]
                a_i = ans[i] if isinstance(ans[i], list) else [ans[i]]
                a_i = [str(x).strip() for x in a_i if str(x).strip()]
                if qs[i] and a_i:
                    cur_i.append(qs[i])
                    cur_a.append(a_i)
            if len(cur_i) > len(best_intents):
                best_intents, best_answers = cur_i, cur_a

        if best_intents:
            intents, answers = _dedup_intents(best_intents, best_answers)
        else:
            # no valid multipleQAs -> single-answer question
            intents = [item['question']]
            sa = []
            for k, t in enumerate(types):
                if t == 'singleAnswer' and k < len(answer_per_round):
                    raw = answer_per_round[k]
                    if isinstance(raw, list):
                        sa.extend(str(x).strip() for x in raw if str(x).strip())
            answers = [sa] if sa else [[]]

        is_ambiguous = len(intents) >= 2
        processed.append({
            'id': item['id'],
            'question': item['question'],
            'is_ambiguous': is_ambiguous,
            'ambiguity_level': len(intents),
            'ground_truth_intents': intents,
            'ground_truth_answers': answers,
            'viewed_doc_titles': item.get('viewed_doc_titles', [])
        })
    return processed


def split_by_ambiguity_level(data: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group by ambiguity level.

    Returns:
        {
            'clear': [...],   # 1 intent
            'light': [...],   # 2 intents
            'heavy': [...]    # 3+ intents
        }
    """
    clear = []
    light = []
    heavy = []

    for item in data:
        level = item['ambiguity_level']
        if level == 1:
            clear.append(item)
        elif level == 2:
            light.append(item)
        else:
            heavy.append(item)

    return {
        'clear': clear,
        'light': light,
        'heavy': heavy
    }


def load_json(file_path: str):
    """Load JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, file_path: str, indent: int = 2):
    """Save JSON."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    print(f"[ok] saved: {file_path}")
