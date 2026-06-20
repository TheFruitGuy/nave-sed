![NAVE Logo](nave.svg)

# NAVE -- Normalized, Adaptive Conformer for Whale Vocalization-Event Detection

Clean, hardcoded implementation of the locked conformer recipe for BioDCASE 2026
Task 2 (3-class Antarctic baleen-whale SED: BMABZ / D / BP). Single-model dev
macro F1 = 0.495; multi-seed probability ensemble on top.

- **N**ormalized -- fixed PCEN channel (per-bin AGC; recovers background-buried D downsweeps)
- **A**daptive -- frequency-dynamic (FDY) convolutions in the stem
- Conformer backbone (macaron FFN / RoPE MHSA / wide depthwise conv k=129)

## Files

This pipeline is **self-contained in four files** (plus this README). Drop them
into any repo and training runs with nothing else from the original project.

| file | role |
|------|------|
| `nave_config.py`   | single hardcoded source of truth (recipe constants + class helpers) |
| `nave_features.py` | 4-ch STFT + PCEN front end (parameter-free) |
| `nave_model.py`    | `NAVE` architecture + checkpoint loaders |
| `nave_train.py`    | training entry point **and** the full data / post-processing / metrics / threshold-tuning pipeline |

`nave_train.py` imports only `nave_config`, `nave_features`, `nave_model` and
third-party packages — the entire data loader, per-epoch negative resampling,
event-level post-processing, metrics, per-class threshold tuner, EMA, optimiser
and validation paths are bundled inside it.

## Requirements

Python ≥ 3.10 and:

```bash
pip install torch numpy pandas scipy soundfile tqdm
```

`tqdm` is optional (a no-op fallback is used if missing). Parquet caching of the
file manifest / annotations is used when a parquet engine (e.g. `pyarrow`) is
installed and silently skipped otherwise.

## Data layout

Point `cfg.DATA_ROOT` (in `nave_config.py`) at the BioDCASE development set:

```
DATA_ROOT/
  train/      annotations/{dataset}.csv   audio/{dataset}/*.wav
  validation/ annotations/{dataset}.csv   audio/{dataset}/*.wav
```

Audio is 250 Hz mono WAV; annotation CSVs carry `start_datetime`, `end_datetime`,
`annotation` (the 7 fine call types). The train/val site lists live in
`nave_config.py`.

## Recipe (all in `nave_config.py`)

STFT SR=250 / N_FFT=256 / hop=5 (129 bins, 20 ms). 4 channels [demeaned |S|,
cos phi, sin phi, PCEN]. FDY on filterbank+feat0 (basis 4). d_model 128, 4 heads,
4 layers, ffn x4, dropout 0.1, depthwise conv k=129. RAdam const LR 5e-5, wd 1e-3,
EMA 0.999, 40 epochs, batch 32, neg-ratio 1.0, per-epoch negative resampling.
Post: 500 ms median smooth -> tuned per-class thresholds -> 0.5 s merge gap ->
0.5-30 s duration filter. Exactly 2,693,499 parameters.

## Usage

```bash
# train one seed (vary --seed to build an ensemble of checkpoints)
CUDA_VISIBLE_DEVICES=0 python nave_train.py --seed 42 --tune-workers 20
```

Each run writes `runs/nave_s<seed>_<timestamp>/nave_best.pt` (best tuned macro)
and `nave_epoch_NN.pt`. Every checkpoint stores `model_state_dict`, the tuned
per-class `thresholds`, `macro_f1`, `epoch` and `seed`.

Load a trained checkpoint anywhere:

```python
from nave_model import NAVE
model = NAVE()
ckpt = model.load_checkpoint("runs/nave_s42_.../nave_best.pt")   # dict with thresholds, etc.
model.eval()
```

`nave_train.py` also exposes the reusable building blocks (`NAVEFeatureExtractor`
is in `nave_features.py`; the data loader `build_val_segments` / `WhaleDataset` /
`collate_fn`, the post-processing `postprocess_predictions`, the metric
`compute_metrics`, and the tuner `tune_thresholds_per_class`) if you want to
write your own evaluation or ensemble script on top — import them straight from
`nave_train`.

## Checkpoint compatibility

`nave_model.py` keeps a loader for checkpoints trained by the previous
`train_phase13r` harness: `NAVE.from_legacy_checkpoint(path)` remaps the old
module names (`_inner.* -> stem.*`, `_proj -> proj`, `classifier -> head`),
drops the dead WhaleVAD BiLSTM keys, and surfaces the stored tuned thresholds.
Verified: 202/202 keys, 0 missing / 0 unexpected, output byte-identical to the
existing model (max abs diff 0.0), param count exactly 2,693,499. Native NAVE
checkpoints load with `NAVE().load_checkpoint(path)`.

```python
from nave_model import NAVE
m, meta = NAVE.from_legacy_checkpoint("runs/phase13r_3c_s42_<ts>/phase13r_best.pt")
print("loaded NAVE", sum(p.numel() for p in m.parameters()), meta)
```

## Logging

Training progress -- per-epoch train/val loss, per-class P/R/F1, the tuned
per-class thresholds and the running best macro F1 -- is printed to stdout.
There is no external experiment tracker.
