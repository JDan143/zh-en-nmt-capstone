# SpeechBridge — From-Scratch NMT Architecture Comparison (ZH ↔ EN)

A comparison of five neural MT architectures for bidirectional Mandarin ↔ English
translation, each implemented from scratch in PyTorch and trained on free cloud
GPUs (Colab / Kaggle).

Ground rules we set for a fair comparison: no pretrained LMs, no translation APIs,
and one shared tokenizer, dataset, preprocessing pipeline, training loop, and eval
suite across all five. Runs are reproducible (fixed seeds, pinned deps, versioned
checkpoints, logged metrics).

All five expose the same interface (`encode` + `_decode_full` in
`models/common.Seq2SeqBase`), so the training loop, beam search, and metrics are
shared and only the model itself changes.

| # | Model | Key idea |
|---|-------|----------|
| 1 | GRU Seq2Seq + Bahdanau attention | additive attention, scheduled sampling |
| 2 | LSTM Seq2Seq + Luong attention | dot/general/concat + input-feeding |
| 3 | ConvS2S | GLU conv blocks, causal padding, multi-step attention |
| 4 | Transformer (vanilla) | sinusoidal PE, Noam warmup, beam 5 |
| 5 | Improved Transformer | relative PE (Shaw, k=16) + label smoothing 0.1 |

Our hypothesis: Arch 5 beats Arch 4 by at least 0.5 BLEU in both directions with no
extra compute. We test it with a paired bootstrap (`evaluate.py --significance`).

---

## Data and tools

Library versions are pinned in `requirements.txt`. We use sentencepiece 0.2.1 and
sacrebleu 2.6.0 (the `flores200`/spBLEU tokenizer needs sacrebleu ≥ 2.3.0). torch is
left unpinned because Colab and Kaggle ship their own CUDA build and forcing a wheel
tends to break GPU support.

Metrics come from sacrebleu: `BLEU(tokenize='zh')` for Chinese output and `'13a'` for
English, chrF++ as `CHRF(word_order=2)`, `TER()`, and spBLEU as
`BLEU(tokenize='flores200')`.

Datasets:
- OpenSubtitles2018 ZH-EN, pulled as an OPUS moses zip
  (`object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-zh_cn.txt.zip`). We download
  it directly because the HF `Helsinki-NLP/open_subtitles` loader is script-based and
  modern `datasets` blocks it.
- News Commentary v18 from statmt (`.tsv.gz`).
- WMT newstest as the test set, fetched with `sacrebleu -t wmt19 -l zh-en --echo src/ref`.
- FLORES+ (`openlanguagedata/flores_plus`, `devtest` split) for the held-out benchmark.
  Eval only — never trained on. It uses the Mandarin code `cmn_Hans` (the older
  facebook/flores used `zho_Hans`).

## A few choices worth noting

- **Arch 3 optimizer.** ConvS2S (Gehring et al., 2017) was trained with NAG. The
  settings lr 0.25 / momentum 0.99 are NAG numbers; under an Adam-family optimizer
  lr 0.25 just diverges, since Adam normalizes the gradient and the step stays ~lr
  regardless of clipping. So Arch 3 uses `SGD(..., nesterov=True)`. It also trains in
  fp32 (`amp_safe=False`) — its stacked GLU/residual blocks overflow fp16 to NaN.
- **Length filter** is measured before subwording: English by whitespace tokens,
  Chinese by characters (Chinese isn't pre-segmented). The subword-length cap is
  reapplied after SentencePiece.
- **Arch 5** uses relative self-attention and drops absolute PE, the standard Shaw
  setup. See `transformer_common.py`.
- **Paired bootstrap** is implemented locally rather than via a sacrebleu class whose
  name has moved around between versions.

---

## Install

```bash
pip install -r requirements.txt
# On Colab/Kaggle, keep the platform torch, don't reinstall it.
```

Paths auto-detect Colab (`/content`), Kaggle (`/kaggle/working`), and local, and can
be overridden with `SPEECHBRIDGE_ROOT`, `_DATA`, `_CKPT`, `_RESULTS`.

## Run — smoke test first

Run the whole path on a tiny subset before committing to a long run:

```bash
export SPEECHBRIDGE_SMOKE=1            # tiny data, tiny model, ~2 epochs
python data/preprocess.py             # download + clean + split (smoke skips langid)
python data/train_tokenizer.py        # shared SentencePiece BPE (small vocab in smoke)
python train.py --arch arch1_gru      # ~2 epochs, ≤30 batches/epoch
python evaluate.py --ckpts $SPEECHBRIDGE_ROOT/checkpoints/arch1_gru_seed191_best.pt \
                   --beam 1 --max-eval 50
```

Once that's clean, scale up:

```bash
unset SPEECHBRIDGE_SMOKE
python data/preprocess.py
python data/train_tokenizer.py
for A in arch1_gru arch2_lstm arch3_convs2s arch4_transformer arch5_improved_transformer; do
  for S in 191 1604 4090; do python train.py --arch $A --seed $S; done
done
```

## Evaluate (mean ± std over 3 seeds, WMT + FLORES+)

```bash
python evaluate.py --ckpts arch4_transformer_seed191_best.pt \
                           arch4_transformer_seed1604_best.pt \
                           arch4_transformer_seed4090_best.pt --beam 5
# Arch5 vs Arch4 significance (paired bootstrap, n=1000, seed=123):
python evaluate.py --significance arch4_transformer_seed191_best.pt \
                                  arch5_improved_transformer_seed191_best.pt
```

## Interactive demo

```bash
python translate.py --ckpt <best.pt> --direction zh-en --text "你好，请问洗手间在哪里？"
```

---

## Results table (fill from `results/eval_*.json`)

WMT newstest (zh→en / en→zh), mean ± std over seeds {191,1604,4090}:

| Model | Params (M) | BLEU ↑ | chrF++ ↑ | TER ↓ | PPL ↓ | tok/s ↑ | BLEU/M ↑ |
|-------|-----------:|-------:|---------:|------:|------:|--------:|---------:|
| Arch 1 GRU + Bahdanau |  |  |  |  |  |  |  |
| Arch 2 LSTM + Luong |  |  |  |  |  |  |  |
| Arch 3 ConvS2S |  |  |  |  |  |  |  |
| Arch 4 Transformer |  |  |  |  |  |  |  |
| Arch 5 Improved Transformer |  |  |  |  |  |  |  |

FLORES+ devtest (zh→en / en→zh): spBLEU, chrF++ (same layout).

---

## Repository layout

```
zh-en-nmt-capstone/
├── requirements.txt          pinned deps
├── config.py                 central config (per-arch hyperparams, paths, smoke profile)
├── utils.py                  seeding, metrics CSV, VRAM/timing
├── data/
│   ├── preprocess.py         download + cleaning + split (SEED=123)
│   ├── train_tokenizer.py    shared SentencePiece BPE (32k)
│   └── dataset.py            bidirectional tagging, collate, max-tokens sampler
├── models/
│   ├── common.py             Seq2SeqBase, masks, label smoothing, Noam, beam/greedy
│   ├── transformer_common.py from-scratch MHA (+ Shaw relative), enc/dec layers
│   ├── arch1_gru.py … arch5_improved_transformer.py
├── train.py                  training harness (accumulation, clip, scheduler, early stop)
├── evaluate.py               BLEU/chrF++/TER/PPL/efficiency + WMT + FLORES+ + bootstrap
├── translate.py              interactive beam-search demo
└── results/                  metrics_log.csv, checkpoints, eval_*.json
```

## Reproducibility notes
- Seeds: `SEED=123` everywhere; `EVAL_SEEDS=[191,1604,4090]`; `cudnn.deterministic=True`.
- The same `bpe_zh_en.model` is reused by every architecture.
- Each run writes `{arch}_seed{seed}_best.pt` and appends a row to `results/metrics_log.csv`.

## Performance notes
These keep the 5-arch × multi-seed plan feasible on a free T4. None of them change
results — the incremental decoders match the full recompute (identical tokens;
logits agree to fp32 rounding).
- **Incremental decoding** for every architecture: KV cache (Transformer), carried
  hidden state (RNN), cached causal buffers (ConvS2S). Each step is O(L) instead of
  re-running the whole prefix. The full-recompute path stays as a correctness fallback.
- **Mixed precision (AMP)** in training and eval, on CUDA only (no-op on CPU): roughly
  2× speed and half the VRAM on a T4. ConvS2S is the exception and runs fp32 (see above).
- **Lazy tokenization**: raw text is stored and tokenized on access, so we don't
  eagerly tokenize ~3M sequences at startup.
- **Eval memory**: per-seed hypotheses are dropped unless the significance test needs them.

## Training length and learning rate
- Epoch count isn't fixed. Training runs until dev loss stalls for
  `EARLY_STOP_PATIENCE=10` epochs. `EPOCH_CAP=200` in `config.py` is only a runaway
  guard, not a target — raise it if a model is still improving at the cap.
- Learning rate adapts separately from stopping. Transformers use Noam
  (warmup → inverse-sqrt decay); RNN/Conv use ReduceLROnPlateau (×0.5 after 3 stalled
  epochs, `min_lr=1e-6`). Reduction patience (3) is shorter than early-stop patience
  (10) so the LR drops before training halts. Per-epoch LR is logged to
  `metrics_log.csv`. Early stopping ends training; the scheduler rescales the step.
  Both are on. See `train.py` (`bad_epochs`/`patience`) and
  `models/common.py:build_optimizer_scheduler`.
