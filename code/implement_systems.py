"""
RAG system implementations (IVF + CPU; the RIV clarification path emits multiple answers).
"""
import sys
sys.path.append('.')

import re
import time
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from config import config
from utils.xinference_client import XInferenceClient
from utils.metrics import semantic_dedup, dynamic_threshold_from_retrieved_docs
from utils.answer_squeeze import squeeze_ambigqa_answer, AMBIGQA_ANSWER_RULES


def _build_context(docs):
    return "\n\n".join([f"[{i+1}] {d['text'][:400]}" for i, d in enumerate(docs)])

def _answer_prompt(query, ctx):
    return (f"{AMBIGQA_ANSWER_RULES}\n\nPassages:\n{ctx}\n\n"
            f"Question: {query}\n\nYour answer (English, minimal):")


class EnhancedRetriever:
    """Two-stage retriever (IVF + CPU)."""

    def __init__(self, xinference_client: XInferenceClient):
        self.xinference = xinference_client
        print("  loading FAISS IVF index (CPU)...")
        t0 = time.time()
        self.index = faiss.read_index(config.FAISS_INDEX_PATH)
        try:
            self.index.nprobe = int(getattr(config, "FAISS_NPROBE", 32))  # IVF only; flat index has no nprobe
        except Exception:
            pass
        print(f"  [ok] index loaded ({time.time()-t0:.1f}s, CPU, ntotal={self.index.ntotal:,})")
        with open(config.METADATA_PATH, 'rb') as f:
            self.passages = pickle.load(f)['passages']
        print(f"  [ok] retriever ready: {len(self.passages):,} passages")

    def retrieve(self, query: str, top_k: int = None, use_rerank: bool = True) -> List[Dict]:
        if top_k is None:
            top_k = int(getattr(config, "TOP_K_RETRIEVE", 5))
        q = np.array(self.xinference.embed([query], model=config.EMBEDDING_MODEL), dtype='float32')
        faiss.normalize_L2(q)
        recall_k = config.TOP_K_RERANK if use_rerank else top_k
        scores, indices = self.index.search(q, recall_k)
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.passages):
                candidates.append({
                    'text': self.passages[idx]['text'],
                    'title': self.passages[idx]['title'],
                    'id': self.passages[idx]['id'],
                    'vector_score': float(score)
                })
        if use_rerank and len(candidates) > top_k:
            try:
                rr = self.xinference.rerank(query, [c['text'] for c in candidates],
                                            model=config.RERANK_MODEL, top_k=top_k)
                return [{**candidates[r['index']], 'rerank_score': r['relevance_score']} for r in rr]
            except Exception:
                return candidates[:top_k]
        return candidates[:top_k]


class VanillaRAG:
    def __init__(self, retriever, xinference_client):
        self.retriever = retriever
        self.xinference = xinference_client

    def answer(self, query: str, top_k: int = None) -> Dict[str, Any]:
        start = time.time()
        if top_k is None:
            top_k = int(getattr(config, "TOP_K_RETRIEVE", 5))
        retrieved = self.retriever.retrieve(query, top_k, use_rerank=True)
        try:
            raw = self.xinference.chat([{"role": "user", "content": _answer_prompt(query, _build_context(retrieved))}],
                                       model=config.LLM_MODEL, temperature=config.TEMPERATURE,
                                       max_tokens=config.MAX_TOKENS)
            ans = squeeze_ambigqa_answer(raw) or "Unable to answer."
        except Exception as e:
            print(f"  [warn] LLM failed: {e}")
            ans = "Unable to answer."
        return {'answer': ans, 'preds': [ans], 'status': 'answered',
                'retrieved': retrieved, 'latency': time.time() - start}


class RIVAgent:
    def __init__(self, retriever, xinference_client):
        self.retriever = retriever
        self.xinference = xinference_client

    def reverse_intent_generation(self, retrieved_docs, n=None):
        if n is None:
            n = int(getattr(config, "N_REVERSE_INTENTS", 3))
        n = max(1, n)
        ctx = "\n\n".join([f"Doc {i+1}: {d['text'][:300]}" for i, d in enumerate(retrieved_docs[:3])])
        lines = "\n".join([f"{i+1}. [Question {i+1}]?" for i in range(n)])
        prompt = (f"Based on these documents, generate {n} different, specific questions "
                  f"users might ask.\n\n{ctx}\n\nGenerate {n} numbered questions "
                  f"(each ending with ?):\n{lines}\n\nQuestions:")
        try:
            resp = self.xinference.chat([{"role": "user", "content": prompt}],
                                        model=config.LLM_MODEL, temperature=config.TEMPERATURE, max_tokens=150)
            qs = [q.strip() for q in re.findall(r'\d+\.\s*([^?\n]+\?)', resp) if len(q.strip()) > 10]
            uniq, seen = [], set()
            for q in qs:
                if q.lower() not in seen:
                    uniq.append(q); seen.add(q.lower())
            if uniq:
                while len(uniq) < n:
                    uniq.append(uniq[-1])
                return uniq[:n]
            return ["What is discussed?"] * n
        except Exception as e:
            print(f"  [warn] reverse generation failed: {e}")
            return ["What is discussed?"] * n

    def compute_similarity(self, query, candidate_intents):
        if not candidate_intents:
            return 0.0, -1
        try:
            embs = np.array(self.xinference.embed([query] + candidate_intents, model=config.EMBEDDING_MODEL))
            sims = cosine_similarity(embs[:1], embs[1:])[0]
            best = int(sims.argmax()); max_sim = float(sims[best])
            n = len(candidate_intents)
            if n > 1:
                pmat = cosine_similarity(embs[1:])
                div = float(np.std([pmat[i, j] for i in range(n) for j in range(i+1, n)]))
            else:
                div = 0.0
            alpha = float(getattr(config, "DIVERSITY_ALPHA", 0.15))
            return max_sim - alpha * div, best
        except Exception as e:
            print(f"  [warn] similarity failed: {e}")
            return 0.0, -1

    def calculate_dynamic_threshold(self, retrieved_docs, base=None):
        if base is None:
            base = float(getattr(config, "BASE_THRESHOLD", 0.42))
        return dynamic_threshold_from_retrieved_docs(
            retrieved_docs, float(base),
            float(getattr(config, "THRESHOLD_ALPHA", 0.10)),
            float(getattr(config, "THRESHOLD_BETA", 0.05)))

    @staticmethod
    def _dedup_answers(answers):
        out, seen = [], set()
        for a in answers:
            k = a.lower().strip()
            if k and k not in seen:
                out.append(a.strip()); seen.add(k)
        return out

    def _answer_one(self, query, top_k):
        docs = self.retriever.retrieve(query, top_k, use_rerank=True)
        raw = self.xinference.chat([{"role": "user", "content": _answer_prompt(query, _build_context(docs))}],
                                   model=config.LLM_MODEL, temperature=config.TEMPERATURE,
                                   max_tokens=config.MAX_TOKENS)
        return squeeze_ambigqa_answer(raw) or ""

    def answer(self, query: str, top_k: int = None) -> Dict[str, Any]:
        start = time.time()
        if top_k is None:
            top_k = int(getattr(config, "TOP_K_RETRIEVE", 5))
        try:
            retrieved = self.retriever.retrieve(query, top_k, use_rerank=True)
            threshold = self.calculate_dynamic_threshold(retrieved, getattr(config, 'BASE_THRESHOLD', 0.42))
            n_intents = int(getattr(config, "N_REVERSE_INTENTS", 3))
            cands = self.reverse_intent_generation(retrieved, n=n_intents)
            cands = semantic_dedup(self.xinference, cands,
                                   sim_threshold=float(getattr(config, "SEMANTIC_DEDUP_THRESHOLD", 0.90)))
            if not cands:
                cands = [query]
            sim_score, best_idx = self.compute_similarity(query, cands)

            if sim_score >= threshold:
                # judged clear -> single answer
                ans = self._answer_one(query, top_k) or "Unable to answer."
                return {'answer': ans, 'preds': [ans], 'status': 'answered',
                        'similarity_score': sim_score, 'dynamic_threshold': threshold,
                        'candidate_intents': cands, 'retrieved': retrieved,
                        'latency': time.time() - start}
            else:
                # judged ambiguous -> retrieve + answer per intent -> multiple answers
                preds = []
                for intent in cands:
                    try:
                        a = self._answer_one(intent, top_k)
                        if a and a.strip():
                            preds.append(a.strip())
                    except Exception:
                        continue
                preds = self._dedup_answers(preds)
                if not preds:
                    preds = [self._answer_one(query, top_k) or "Unable to answer."]
                answer_text = "\n".join(f"{i+1}. {p}" for i, p in enumerate(preds))
                return {'answer': answer_text, 'preds': preds, 'status': 'clarification',
                        'similarity_score': sim_score, 'dynamic_threshold': threshold,
                        'candidate_intents': cands, 'retrieved': retrieved,
                        'latency': time.time() - start}
        except Exception as e:
            print(f"  [warn] RIV-Agent error: {e}")
            return {'answer': f"Error: {e}", 'preds': [], 'status': 'error',
                    'similarity_score': 0.0,
                    'dynamic_threshold': float(getattr(config, 'BASE_THRESHOLD', 0.42)),
                    'candidate_intents': [], 'retrieved': [], 'latency': time.time() - start}


class HyDERAG:
    def __init__(self, retriever, xinference_client):
        self.retriever = retriever
        self.xinference = xinference_client

    def answer(self, query: str, top_k: int = None) -> Dict[str, Any]:
        start = time.time()
        if top_k is None:
            top_k = int(getattr(config, "TOP_K_RETRIEVE", 5))
        retrieved, ans = [], "Unable to answer."
        try:
            hypo = self.xinference.chat([{"role": "user", "content": (
                f"Write one short English paragraph with facts useful to answer the question.\n\n"
                f"Question: {query}\n\nParagraph:")}],
                model=config.LLM_MODEL, temperature=config.TEMPERATURE,
                max_tokens=min(256, config.MAX_TOKENS + 100))
            hypo = (hypo or "").strip()[:800] or query
            retrieved = self.retriever.retrieve(hypo, top_k, use_rerank=True)
            raw = self.xinference.chat([{"role": "user", "content": _answer_prompt(query, _build_context(retrieved))}],
                                       model=config.LLM_MODEL, temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS)
            ans = squeeze_ambigqa_answer(raw or "") or "Unable to answer."
        except Exception as e:
            print(f"  [warn] HyDE failed: {e}")
        return {'answer': ans, 'preds': [ans], 'status': 'answered',
                'retrieved': retrieved, 'latency': time.time() - start}
