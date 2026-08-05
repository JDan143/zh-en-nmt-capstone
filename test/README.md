# `test/` — Unidirectional ablation (research experiment)

Does training **two separate one-direction models** (EN→ZH and ZH→EN) score higher
**per direction** than the single **bidirectional** model the study built? This folder
answers that, as a controlled comparison.

The question per architecture: is unidirectional-EN→ZH BLEU > bidirectional-EN→ZH BLEU?
And the same for ZH→EN?

---

## What is held IDENTICAL to the study (the fairness checklist)

| Held constant | Value |
|---|---|
| Data | the **same 800k** mixture (OpenSubtitles 800k + News Commentary) — **not** the scaled 3.34M set |
| Tokenizer | the **same joint 32k** `bpe_zh_en.model` — **reused, never retrained** |
| Architecture / params | identical (`arch4_transformer`, `arch5_improved_transformer`) |
| Optimizer / schedule | Adam + Noam warmup, identical |
| Label smoothing, weight tying, beam + length penalty, AMP gating | identical |
| Early stopping | patience 10, no epoch ceiling, auto-adjusting LR — identical |
| Seed | **191**, matching the study's seed-191 numbers (1604 / 4090 can be added later) |
| Test sets | same WMT newstest + FLORES+ devtest |

**The ONLY thing that differs is the training direction.** Any BLEU difference must
therefore come from unidirectional-vs-bidirectional training, not from data, vocabulary,
capacity, or evaluation.

## Why the tokenizer is reused (not rebuilt)

The 32k vocab is **joint** — it already contains both Chinese and English subwords. A
ZH→EN model reads Chinese and writes English; both are in the vocab. A direction-specific
tokenizer would confound the experiment (a BLEU change could then be the vocabulary, not
the direction). A direction-specific-tokenizer variant is a *separate, later* experiment.
(Will test a rebuilt one soon as well)

## Reading the result

For each architecture, place the ablation number beside the study's bidirectional
seed-191 number for the same direction:

| Arch | Direction | Bidirectional BLEU (study, seed 191) | Unidirectional BLEU (this) | Δ |
|---|---|---|---|---|
| arch4 | EN→ZH | *(from study eval)* | *(eval_arch4_transformer_enzh_seed191.json)* | |
| arch4 | ZH→EN | | | |
| arch5 | EN→ZH | | | |
| arch5 | ZH→EN | | | |

A positive Δ means unidirectional helped that direction; negative means the shared
bidirectional model was as good or better (the common finding at this data scale — one
model, half the storage, comparable quality). Either way it is a defensible, measured
answer to "was the bidirectional design choice justified?"
