"""
AmbigQA evaluation helper: squeeze long LLM outputs into short answers to raise
overlap with the token-F1 references. Does not change facts; only truncates and
strips common prefixes.
"""

import re
from typing import List

AMBIGQA_ANSWER_RULES = (
    "Rules: Reply in English only. Output ONLY the minimal factual answer "
    "(a name, date, year, number, or at most 12 words). "
    "No sentences like 'The answer is …'. No explanation beyond the answer itself."
)


def squeeze_ambigqa_answer(text: str, max_words: int = 16) -> str:
    if not text or not str(text).strip():
        return ""
    t = str(text).strip()
    # drop leading chain-of-thought lines
    t = re.split(r"\n\s*\n", t)[0].strip()
    t = t.split("\n")[0].strip()

    lower = t.lower()
    for prefix in (
        "answer:",
        "a:",
        "final answer:",
        "the answer is",
    ):
        if lower.startswith(prefix):
            t = t[len(prefix) :].strip()
            lower = t.lower()

    t = t.strip(" \t\"'`*_")

    # if the text is "explanation + trailing entity", try the last sentence (still word-limited)
    if len(t.split()) > max_words and "." in t:
        tail = t.split(".")[-1].strip()
        if tail and len(tail.split()) <= max_words + 4:
            t = tail

    words = t.split()
    if len(words) > max_words:
        t = " ".join(words[:max_words])
    else:
        t = " ".join(words)
    return t.strip()


def squeeze_many(texts: List[str], max_words: int = 16) -> List[str]:
    return [squeeze_ambigqa_answer(x, max_words=max_words) for x in texts]
