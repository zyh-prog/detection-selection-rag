# The SituatedQA evaluation split

SituatedQA (Zhang & Choi, EMNLP 2021) ships one row per *situated instance*: an ambiguous
question, a disambiguated `edited_question` carrying a date or a location, and that
situation's answer. Rows that share an `id` are situations of the same question.

The paper groups those rows by `id` to recover multi-answer ground truth — a question counts
as ambiguous when it has two or more distinct situations — and samples 500 questions with
seed 42, split 100 dev / 400 test.

## Why this directory ships identifiers instead of the data

The grouped file contains SituatedQA's own question and answer text. The upstream release
(<https://github.com/mikejqzhang/SituatedQA>) carries **no licence statement**, so we do not
have clear permission to redistribute its content, and we are not going to assume it. What
is shipped here is only what is ours:

| field | whose |
|---|---|
| `id` | upstream identifier, prefixed `geo_` / `temp_` |
| `source` | `geo` or `temp`, from the source file |
| `split` | **ours** — dev / test assignment, seed 42 |
| `is_ambiguous` | **ours** — derived, `>= 2` distinct situations |
| `ambiguity_level` | **ours** — number of distinct situations |
| `content_sha256_16` | **ours** — a fingerprint so you can verify your rebuild |

No question text, no interpretations, no answers.

## Rebuilding the full split

1. Get the four files from the SituatedQA release (`data/qa_data`):
   `geo.test.jsonl`, `geo.dev.jsonl`, `temp.test.jsonl`, `temp.dev.jsonl`.
2. Put them in `data/situatedqa/raw/`.
3. Run `python code/situatedqa_loader.py`.

It groups, samples with seed 42, and writes `data/situatedqa/situatedqa_annotated.json` plus
the dev/test id lists. The result is deterministic: the same upstream files give the same
500 questions in the same order.

## Verifying your rebuild

Each row here carries `content_sha256_16`, the first 16 hex characters of the SHA-256 of

```python
json.dumps({"q": question, "i": ground_truth_intents, "a": ground_truth_answers},
           ensure_ascii=False, sort_keys=True)
```

for that question. Recompute it over your rebuilt records and compare. If every fingerprint
matches, your split is identical to the one used in the paper — verified without either of us
having to move the upstream text around.

## What the split should look like

| | |
|---|---:|
| questions | 500 |
| ambiguous / clear | 350 / 150 |
| dev / test | 100 / 400 |
| geo / temp | 100 / 400 |

The paper's SituatedQA results are all on the 400-question test portion.
