"""situatedqa_loader.py — a third, structurally independent ambiguity dataset.

SituatedQA (Zhang & Choi 2021): under-specified questions whose answer depends on
a temporal or geographic *situation*. Genuinely independent of AmbigQA/ASQA
(entity/event ambiguity from NQ): here ambiguity is context-dependence.

Each raw row = one situated instance: `question` (ambiguous), `edited_question`
(disambiguated with date/location), `answer` (that situation's answer), sharing
an `id` across situations. We group by `id` to recover the multi-intent structure,
matching the ASQA schema produced by asqa_loader.py:
  {id, question, is_ambiguous, ambiguity_level, ground_truth_intents,
   ground_truth_answers, viewed_doc_titles}

Raw jsonl already fetched to data/situatedqa/raw/ from
https://github.com/mikejqzhang/SituatedQA (data/qa_data).
"""
import sys, json, random
sys.path.append(".")
from pathlib import Path
from collections import OrderedDict
from config import config

RAW = Path("data/situatedqa/raw")
OUT = "data/situatedqa/situatedqa_annotated.json"
SPLIT = "data/situatedqa/riv_eval_split.json"
FILES = ["geo.test.jsonl", "geo.dev.jsonl", "temp.test.jsonl", "temp.dev.jsonl"]
N = int(getattr(config, "EVAL_SAMPLE_SIZE", 500))
SEED, DEV_SIZE = 42, 100


def norm(s):
    return " ".join(str(s).strip().split())


def main():
    Path("data/situatedqa").mkdir(parents=True, exist_ok=True)
    groups = OrderedDict()   # id -> {question, source, intents{edited_q: set(answers)}}
    for fn in FILES:
        p = RAW / fn
        if not p.exists():
            print("  [warn] missing", p); continue
        src = "geo" if fn.startswith("geo") else "temp"
        for line in open(p):
            if not line.strip():
                continue
            ex = json.loads(line)
            qid = f"{src}_{ex['id']}"
            q = norm(ex.get("question", ""))
            eq = norm(ex.get("edited_question", "") or ex.get("question", ""))
            ans = [norm(a) for a in (ex.get("answer") or []) if norm(a)]
            if not q or not eq or not ans:
                continue
            g = groups.setdefault(qid, {"question": q, "source": src, "intents": OrderedDict()})
            g["intents"].setdefault(eq, [])
            for a in ans:
                if a not in g["intents"][eq]:
                    g["intents"][eq].append(a)

    out = []
    for qid, g in groups.items():
        intents = list(g["intents"].keys())
        answers = [g["intents"][i] for i in intents]
        if not intents:
            continue
        out.append({
            "id": qid,
            "question": g["question"],
            "source": g["source"],
            "is_ambiguous": len(intents) >= 2,
            "ambiguity_level": len(intents),
            "ground_truth_intents": intents,
            "ground_truth_answers": answers,
            "viewed_doc_titles": [],
        })

    rng = random.Random(SEED)
    # keep the ambiguity distribution informative: prefer multi-situation items but keep some singletons
    amb = [r for r in out if r["is_ambiguous"]]
    sng = [r for r in out if not r["is_ambiguous"]]
    rng.shuffle(amb); rng.shuffle(sng)
    # target ~70% ambiguous to mirror an ambiguity benchmark, capped at N
    n_amb = min(len(amb), int(N * 0.7))
    n_sng = min(len(sng), N - n_amb)
    sample = amb[:n_amb] + sng[:n_sng]
    rng.shuffle(sample)
    if len(sample) > N:
        sample = sample[:N]

    json.dump(sample, open(OUT, "w"), ensure_ascii=False, indent=2)
    ids = [r["id"] for r in sample]
    random.Random(SEED).shuffle(ids)
    json.dump({"dev": ids[:DEV_SIZE], "test": ids[DEV_SIZE:]}, open(SPLIT, "w"), indent=2)

    n_amb_final = sum(r["is_ambiguous"] for r in sample)
    geo = sum(r["source"] == "geo" for r in sample)
    levels = {}
    for r in sample:
        levels[r["ambiguity_level"]] = levels.get(r["ambiguity_level"], 0) + 1
    print(f"[ok] SituatedQA: {len(sample)} questions "
          f"(ambiguous {n_amb_final}, clear {len(sample)-n_amb_final}; geo {geo}, temp {len(sample)-geo}) "
          f"-> {OUT} + {SPLIT}")
    print("  ambiguity_level histogram:", dict(sorted(levels.items())))
    print("  raw groups available:", len(out), f"(ambiguous {len(amb)}, singleton {len(sng)})")
    e = sample[0]
    print("  sample[0]:", e["question"], "| intents:", e["ground_truth_intents"][:2],
          "| answers:", e["ground_truth_answers"][:2])


if __name__ == "__main__":
    main()
