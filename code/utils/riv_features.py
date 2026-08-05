"""
utils/riv_features.py
Unified RIV feature-extraction module.

Every expensive operation runs once per question:
- original-question retrieval + direct answer
- reverse-intent generation
- an answer for each generated intent
- gold-intent oracle answers
- answer embeddings
"""

import hashlib
import pickle
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from config import config
from utils.answer_squeeze import AMBIGQA_ANSWER_RULES, squeeze_ambigqa_answer
from utils.riv_policy import _norm_answer, unique_nonempty_answers


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


def build_context(docs: Sequence[Dict], max_chars: int = 360) -> str:
    parts = []
    for i, doc in enumerate(docs):
        title = str(doc.get("title", "") or "")
        text = str(doc.get("text", "") or "")[:max_chars]
        if title:
            parts.append(f"[{i + 1}] {title}\n{text}")
        else:
            parts.append(f"[{i + 1}] {text}")
    return "\n\n".join(parts)


def answer_prompt(question: str, ctx: str) -> str:
    return (
        f"{AMBIGQA_ANSWER_RULES}\n\n"
        f"Passages:\n{ctx}\n\n"
        f"Question: {question}\n\n"
        f"Your answer (English, minimal):"
    )


class RIVFeatureExtractor:
    def __init__(self, retriever, xinference_client, use_rerank: bool = True):
        self.retriever = retriever
        self.xinference = xinference_client
        self.use_rerank = use_rerank

    def retrieve(self, query: str, top_k: int = None):
        if top_k is None:
            top_k = int(getattr(config, "TOP_K_RETRIEVE", 8))
        return self.retriever.retrieve(query, top_k=top_k, use_rerank=self.use_rerank)

    def answer_one(self, question: str, top_k: int = None) -> str:
        docs = self.retrieve(question, top_k=top_k)
        raw = self.xinference.chat(
            [{"role": "user", "content": answer_prompt(question, build_context(docs))}],
            model=config.LLM_MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )
        return squeeze_ambigqa_answer(raw or "")

    def reverse_intents(self, query: str, retrieved_docs: Sequence[Dict], n: int = None) -> List[str]:
        if n is None:
            n = int(getattr(config, "N_REVERSE_INTENTS", 4))
        n = max(1, int(n))

        ctx = build_context(retrieved_docs[:8], max_chars=300)

        titles = []
        seen_titles = set()
        for doc in retrieved_docs[:12]:
            title = str(doc.get("title", "") or "").strip()
            key = title.lower()
            if title and key not in seen_titles:
                seen_titles.add(key)
                titles.append(title)

        title_text = "\n".join(f"- {t}" for t in titles[:12]) or "(none)"

        prompt = (
            "You are resolving AmbigQA-style ambiguous questions.\n"
            "Your job is NOT to paraphrase. Your job is to produce disambiguated factual questions.\n\n"
            "A question is ambiguous when a short mention can refer to multiple entities, works, people, "
            "organizations, places, events, dates, or versions, and those interpretations can have different answers.\n\n"
            "Rules:\n"
            "1. Generate distinct interpretations of the ORIGINAL QUESTION.\n"
            "2. Each line must be a standalone English question.\n"
            "3. Prefer replacing the ambiguous mention with a specific entity/title/person suggested by retrieved titles.\n"
            "4. If the question is clear, output one canonical question plus conservative near-duplicates only if needed.\n"
            "5. Do not ask broader related questions; keep the original relation/time/property unchanged.\n"
            "6. The interpretations should be likely to produce different short answers.\n\n"
            "Examples:\n"
            "Original: When was Apple founded?\n"
            "1. When was Apple Inc. founded?\n"
            "2. When was Apple Records founded?\n"
            "3. When was Apple Corps founded?\n\n"
            "Original: Who played Lincoln?\n"
            "1. Who played Abraham Lincoln in the 2012 film Lincoln?\n"
            "2. Who played Lincoln in the television series The 100?\n"
            "3. Who played Lincoln in Bill & Ted's Excellent Adventure?\n\n"
            f"ORIGINAL QUESTION:\n{query}\n\n"
            f"RETRIEVED TITLES:\n{title_text}\n\n"
            f"PASSAGES:\n{ctx}\n\n"
            f"Return exactly {n} numbered questions."
        )

        try:
            resp = self.xinference.chat(
                [{"role": "user", "content": prompt}],
                model=config.LLM_MODEL,
                temperature=config.TEMPERATURE,
                max_tokens=260,
            )
        except Exception:
            return [query] * n

        out, seen = [], set()
        for line in str(resp or "").splitlines():
            line = re.sub(r"^\s*[-*]?\s*\d+[\).:-]\s*", "", line.strip()).strip()
            if not line:
                continue
            if "?" not in line:
                line = line.rstrip(".") + "?"
            if len(line.split()) < 4:
                continue
            key = line.lower()
            if key not in seen:
                seen.add(key)
                out.append(line)

        if not out:
            out = [query]
        return out[:n]

    def intent_features(self, query: str, intents: Sequence[str]) -> Dict[str, float]:
        if not intents:
            return {"max_sim": 0.0, "eff_sim": 0.0, "intent_diversity": 0.0, "intent_pair_mean": 0.0}

        try:
            embs = np.asarray(
                self.xinference.embed([query, *intents], model=config.EMBEDDING_MODEL),
                dtype=np.float32,
            )
            sims = cosine_similarity(embs[:1], embs[1:])[0]
            max_sim = float(np.max(sims))

            if len(intents) > 1:
                pair = cosine_similarity(embs[1:])
                vals = [float(pair[i, j]) for i in range(len(intents)) for j in range(i + 1, len(intents))]
                div = float(np.std(vals)) if vals else 0.0
                mean_pair = float(np.mean(vals)) if vals else 1.0
            else:
                div = 0.0
                mean_pair = 1.0

            alpha = float(getattr(config, "DIVERSITY_ALPHA", 0.15))
            return {
                "max_sim": max_sim,
                "eff_sim": float(max_sim - alpha * div),
                "intent_diversity": div,
                "intent_pair_mean": mean_pair,
            }
        except Exception:
            return {"max_sim": 0.0, "eff_sim": 0.0, "intent_diversity": 0.0, "intent_pair_mean": 0.0}

    def answer_embeddings(self, answers: Sequence[str]) -> Dict[str, List[float]]:
        answers = unique_nonempty_answers(answers)
        if not answers:
            return {}

        try:
            embs = self.xinference.embed(answers, model=config.EMBEDDING_MODEL)
            return {_norm_answer(a): e for a, e in zip(answers, embs)}
        except Exception:
            return {}

    def extract_one(self, sample: Dict, top_k: int = None, collect_oracle: bool = True) -> Dict:
        start = time.time()
        query = sample["question"]

        retrieved = self.retrieve(query, top_k=top_k)
        direct_answer = self.answer_one(query, top_k=top_k)

        n_intents = int(getattr(config, "N_REVERSE_INTENTS", 4))
        generated_intents = self.reverse_intents(query, retrieved, n=n_intents)

        intent_answers = []
        for intent in generated_intents:
            try:
                if intent.strip().lower() == query.strip().lower():
                    intent_answers.append(direct_answer)
                else:
                    intent_answers.append(self.answer_one(intent, top_k=top_k))
            except Exception:
                intent_answers.append("")

        gt_intents = sample.get("ground_truth_intents") or []
        oracle_answers = []
        if collect_oracle:
            for intent in gt_intents:
                try:
                    oracle_answers.append(self.answer_one(intent, top_k=top_k))
                except Exception:
                    oracle_answers.append("")

        all_answers = [direct_answer, *intent_answers, *oracle_answers]

        return {
            "id": str(sample["id"]),
            "question": query,
            "is_ambiguous": bool(sample.get("is_ambiguous", False)),
            "ambiguity_level": int(sample.get("ambiguity_level", 1)),
            "ground_truth_answers": sample.get("ground_truth_answers", []),
            "ground_truth_intents": gt_intents,
            "gt_answers": sample.get("ground_truth_answers", []),
            "gt_intents": gt_intents,
            "retrieved": [
                {
                    "id": d.get("id"),
                    "title": d.get("title"),
                    "vector_score": d.get("vector_score"),
                    "rerank_score": d.get("rerank_score"),
                }
                for d in retrieved
            ],
            "generated_intents": generated_intents,
            "interpretations": generated_intents,
            "direct_answer": direct_answer,
            "intent_answers": intent_answers,
            # "oracle_intent_answers" = the Gold-interpretation answers (paper term);
            # cache key kept as-is so existing .pkl caches stay loadable.
            "oracle_intent_answers": oracle_answers,
            "ans_texts": [direct_answer, *intent_answers],
            "answer_embeddings": self.answer_embeddings(all_answers),
            "intent_features": self.intent_features(query, generated_intents),
            "latency_s": time.time() - start,
        }

    def extract(self, sample: Dict, top_k: int = None, collect_oracle: bool = True) -> Dict:
        return self.extract_one(sample, top_k=top_k, collect_oracle=collect_oracle)


def make_split(samples: Sequence[Dict], dev_size: int, seed: int = 42) -> Dict[str, List[str]]:
    ids = [str(s["id"]) for s in samples]
    import random
    rng = random.Random(seed)
    rng.shuffle(ids)
    return {"dev": ids[:dev_size], "test": ids[dev_size:]}


def cache_key_for_config() -> str:
    parts = [
        str(getattr(config, "LLM_MODEL", "")),
        str(getattr(config, "EMBEDDING_MODEL", "")),
        str(getattr(config, "RERANK_MODEL", "")),
        str(getattr(config, "TOP_K_RETRIEVE", "")),
        str(getattr(config, "TOP_K_RERANK", "")),
        str(getattr(config, "N_REVERSE_INTENTS", "")),
        str(getattr(config, "TEMPERATURE", "")),
        str(getattr(config, "MAX_TOKENS", "")),
        repr(getattr(config, "LLM_CHAT_EXTRA_BODY", None)),     # include the thinking switch in the key
    ]
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:12]


def collect_cache(extractor, samples, ids, cache_path, refresh=False):
    cache = load_pickle(cache_path, default={})
    sample_by_id = {str(s["id"]): s for s in samples}
    ids = [str(x) for x in ids]
    meta = cache.setdefault("_meta", {})
    new_key = cache_key_for_config()
    old_key = meta.get("config_key")
    if old_key and old_key != new_key:
        print(f"  [warn] config changed ({old_key}->{new_key}); clearing old records and re-collecting")
        cache["records"] = {}
    meta["config_key"] = new_key
    # provenance stamp so each cache records which model/settings produced it
    meta["llm_model"]  = str(getattr(config, "LLM_MODEL", ""))
    meta["max_tokens"] = int(getattr(config, "MAX_TOKENS", 0) or 0)
    meta["extra_body"] = repr(getattr(config, "LLM_CHAT_EXTRA_BODY", None))
    records = cache.setdefault("records", {})
    for sid in tqdm(ids, desc=f"Collect cache {Path(cache_path).stem}"):
        if not refresh and sid in records:
            continue
        if sid not in sample_by_id:
            continue
        records[sid] = extractor.extract_one(sample_by_id[sid], collect_oracle=True)
        if True:
            save_pickle(cache, cache_path)
    save_pickle(cache, cache_path)
    return cache