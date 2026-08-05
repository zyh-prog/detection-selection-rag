# Prompts

All four prompts used in the paper, verbatim. Every call runs at temperature 0.0 with one
sample; the answer calls cap generation at 64 tokens and interpretation generation at 260.

The source of truth is the code: `code/utils/answer_squeeze.py`,
`code/utils/riv_features.py` and `code/verify_candidates.py`. This file is the same text
laid out for reading.

---

## 1. Answer generation

Used for the direct answer and for the answer to each generated interpretation. `{ctx}` is
the retrieved passage block, `{question}` is the question being answered.

```
Rules: Reply in English only. Output ONLY the minimal factual answer (a name, date, year,
number, or at most 12 words). No sentences like 'The answer is …'. No explanation beyond
the answer itself.

Passages:
{ctx}

Question: {question}

Your answer (English, minimal):
```

The passage block is built from the top retrieved documents, each truncated to 360
characters:

```
[1] {title_1}
{text_1}

[2] {title_2}
{text_2}
...
```

A system message is prepended by the client: *"You are a helpful assistant. Follow the
user's format constraints; answer concisely in English when asked for factual QA."*

## 2. Reverse-intent generation

Produces the candidate interpretations (`n = 4` in all reported runs). `{query}` is the
original question, `{title_text}` a bulleted list of up to 12 distinct retrieved titles,
and `{ctx}` the top-8 passages truncated to 300 characters each.

```
You are resolving AmbigQA-style ambiguous questions.
Your job is NOT to paraphrase. Your job is to produce disambiguated factual questions.

A question is ambiguous when a short mention can refer to multiple entities, works, people,
organizations, places, events, dates, or versions, and those interpretations can have
different answers.

Rules:
1. Generate distinct interpretations of the ORIGINAL QUESTION.
2. Each line must be a standalone English question.
3. Prefer replacing the ambiguous mention with a specific entity/title/person suggested by
   retrieved titles.
4. If the question is clear, output one canonical question plus conservative near-duplicates
   only if needed.
5. Do not ask broader related questions; keep the original relation/time/property unchanged.
6. The interpretations should be likely to produce different short answers.

Examples:
Original: When was Apple founded?
1. When was Apple Inc. founded?
2. When was Apple Records founded?
3. When was Apple Corps founded?

Original: Who played Lincoln?
1. Who played Abraham Lincoln in the 2012 film Lincoln?
2. Who played Lincoln in the television series The 100?
3. Who played Lincoln in Bill & Ted's Excellent Adventure?

ORIGINAL QUESTION:
{query}

RETRIEVED TITLES:
{title_text}

PASSAGES:
{ctx}

Return exactly {n} numbered questions.
```

Post-processing strips list markers, appends a question mark when absent, drops lines under
four words, de-duplicates case-insensitively, and truncates to `n`. If the call fails the
original question is repeated `n` times, so a failure degrades to direct answering rather
than to an empty pool.

## 3. Self-verifier

The prompted selector of Section 5.4. One call per candidate; the reply is read as accepted
iff it starts with `VALID`.

```
Original question:
{question}

Candidate interpretation:
{intent}

Candidate answer:
{answer}

Is the candidate interpretation a valid disambiguation of the original question?

Reply VALID only if:
- it preserves the original asked relation/property;
- it only resolves an ambiguous entity/person/work/time/version;
- it could be one plausible meaning of the original question;
- the answer is factual and not a refusal.

Reply INVALID if:
- it changes the topic;
- it asks about a different show/person/event;
- it is merely related retrieval noise;
- the answer is unknown/not specified/refusal.

Reply one token: VALID or INVALID.
```

## 4. NLI evidence grounding

Not a prompt, but a natural-language-inference classifier,
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`. For candidate answer `c` to question `q`,
each retrieved passage is the premise and the hypothesis is

```
{q} The answer is {c}
```

The candidate is kept iff the top predicted label is *entailment* for at least one passage.
Threshold-free: nothing is tuned on the data. The `+τ` variant of Table 3 instead requires
the entailment probability to exceed a threshold tuned on the dev split only (τ = 0.3).
