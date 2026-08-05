# Understanding the Bottleneck in Multi-Intent RAG: A Detection–Selection Decomposition

Code, prompts, and the SituatedQA evaluation split for the NLPCC 2026 paper.

> Zhong, Y., Ou, W., Zhou, F., Li, W. *Understanding the Bottleneck in Multi-Intent RAG: A
> Detection–Selection Decomposition.* NLPCC 2026, Macau. Springer LNAI. *(to appear)*

## What is here

```
PROMPTS.md                      all four prompts, verbatim
data/situatedqa_split_ids.json  the 500-question SituatedQA split (ids + our annotations)
data/README.md                  how to rebuild the split from the upstream release
code/                           pipeline, reference systems, selectors, analysis, figures
```

## What is *not* here, and why

**The caches.** The paper's numbers are read off cached generations — retrieved passages,
generated interpretations, and model answers for every question in every arm. That is 134 GB,
so it is not distributed. Everything needed to regenerate it is here.

**The retrieval index.** A FAISS IVF index over the 21M-passage DPR Wikipedia dump
(`psgs_w100`). Build it from the public dump with `bge-large-en-v1.5`; see `code/config.py`
for the parameters actually used (top-8 of 16 reranked, `nprobe=32`).

**The upstream datasets.** AmbigQA, ASQA and SituatedQA are the authors' to distribute, not
ours. Get them from their own releases; `data/README.md` explains how our SituatedQA split
is rebuilt from the upstream files.

## Reproduction — please read before comparing numbers

**Regenerating from scratch will not reproduce the reported values to the decimal, and that is
expected.** The pipeline calls an LLM at temperature 0.0, but greedy decoding is not
bit-reproducible across serving stacks: batching, kernel and precision differences all move
individual generations, and a changed generation changes a candidate pool and therefore a
score. In our own testing, moving the identical prompts from one inference engine to another
changed the answer sets on roughly a third of questions while leaving every reported effect
in place.

What *should* reproduce is the structure of the result, since every reported quantity is a
paired difference between arms that share one cached pool:

* the selection gap exceeds the detection gain in all six dataset–model settings;
* the ordering survives the detection-equalized control;
* the three training-free selectors over-accept on clear questions by roughly equal amounts;
* the clear-question fraction predicts the sign of each selector's net gain.

If you regenerate and a *sign* flips, that is worth reporting. If a number moves in the second
decimal, that is the decoding stack.

## Running it

```bash
pip install -r requirements.txt

# 1. rebuild the SituatedQA split from the upstream release (see data/README.md)
python code/situatedqa_loader.py

# 2. collect the pipeline caches (needs a served LLM and a built retrieval index)
python code/run_riv_pipeline.py --collect all

# 3. the decomposition and its detection-equalized control
python code/selection_ci.py
python code/joint_control.py
python code/x1_joint_exact.py

# 4. the realized selectors
python code/verify_candidates.py        # prompted self-verifier
python code/grounding_selector.py       # NLI evidence grounding
python code/conj_tau_experiment.py      # conjunction, and the dev-tuned threshold variant
python code/over_acceptance.py          # blindness gap and the clear/ambiguous drag split

# 5. figures
python code/figures/gen_all_figures.py
```

The model arm is selected by environment variable rather than by editing `config.py`:

```bash
# local server (default)
python code/run_riv_pipeline.py --collect all

# API arm
RIV_LLM_BACKEND=deepseek RIV_LLM_MODEL=deepseek-ai/DeepSeek-V3.2 \
DEEPSEEK_API_KEY=... python code/run_riv_pipeline.py --collect all
```

`DEEPSEEK_API_KEY` is read from the environment and is never written to a file.

## Metric

Set-F1 with optimal (Hungarian) assignment and token-overlap edge weights,
`2S/(|P|+|G|)`. It keeps the form of AmbigQA's official answer-F1 but differs in two places:
the assignment is optimal rather than greedy, and the edge weight is token overlap rather
than exact match. **Absolute values are therefore not comparable with published AmbigQA
scores**; every effect in the paper is a paired difference measured inside this one setup.
Implementation: `code/utils/metrics.py`, `code/selection_ci.py`.

All confidence intervals are paired percentile bootstrap, 10,000 resamples, seed 42.

## Licence

The code in `code/` and the annotations in `data/` are released under the MIT licence (see
`LICENSE`). This covers our own work only. The upstream datasets keep their own terms — see
`data/README.md`.

## Contact

Please open an issue for anything that does not run.
