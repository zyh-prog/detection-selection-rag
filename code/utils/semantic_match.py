# Pairwise semantic-equivalence helper for the soft-branch recompute.
# Uses the bge embedding endpoint only; no FAISS, no LLM.
import os
import requests, numpy as np
from functools import lru_cache

XINF_BASE = os.environ.get("XINFERENCE_BASE_URL", "http://127.0.0.1:9997")
EMBED_MODEL = "bge-large-en-v1.5"          # registered embedder uid
EQUIV_THRESHOLD = 0.75                     # cosine >= this counts as equivalent

def _embed_batch(texts):
    if isinstance(texts, str):
        texts = [texts]
    r = requests.post(f"{XINF_BASE}/v1/embeddings",
                      json={"model": EMBED_MODEL, "input": texts}, timeout=60)
    r.raise_for_status()
    return [np.asarray(d["embedding"], dtype=np.float32) for d in r.json()["data"]]

@lru_cache(maxsize=20000)
def _embed_one(t):
    return _embed_batch(t)[0]

def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

def semantic_score(pred, gold):
    """Cosine in [-1,1] between one predicted and one gold answer string (symmetric, no instruction prefix)."""
    return cosine(_embed_one(pred.strip().lower()), _embed_one(gold.strip().lower()))

def semantic_equiv(pred, gold, thr=EQUIV_THRESHOLD):
    return semantic_score(pred, gold) >= thr
