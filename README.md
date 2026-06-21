<p align="center">
  <img src="nave_logo.svg" alt="NAVE Conformer Wave Logo" width="600">
</p>

# NAVE -- Normalized, Adaptive Conformer for Whale Vocalization-Event Detection

A clean implementation of a conformer model for BioDCASE 2026 Task 2 (3-class
Antarctic baleen-whale sound-event detection: BMABZ / D / BP).

- **N**ormalized -- fixed PCEN channel (per-bin AGC; recovers background-buried D downsweeps)
- **A**daptive -- frequency-dynamic (FDY) convolutions in the stem
- Conformer backbone (macaron FFN / RoPE MHSA / wide depthwise conv)

## Files

The training pipeline is **self-contained in four files**; the optional
evaluation script `nave_evaluate.py` adds a fifth file that depends only on the
four.

| file | role |
|------|------|
| `nave_config.py`   | single source of truth for all constants + class helpers |
| `nave_features.py` | 4-ch STFT + PCEN front end (parameter-free) |
| `nave_model.py`    | `NAVE` architecture + native checkpoint loader |
| `nave_train.py`    | training entry point **and** the full data / post-processing / metrics / threshold-tuning pipeline |
| `nave_evaluate.py` | *optional* — single-checkpoint evaluation (one inference pass + tuned per-class thresholds), imports only from the four files above |

`nave_train.py` imports only `nave_config`, `nave_features`, `nave_model` and
third-party packages — the entire data loader, per-epoch negative resampling,
event-level post-processing, metrics, per-class threshold tuner, EMA, optimiser
and validation paths are bundled inside it. `nave_evaluate.py` reuses those
same helpers via `from nave_train import …`.

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
0.5-30 s duration filter.

## Usage

```bash
# train one seed
CUDA_VISIBLE_DEVICES=0 python nave_train.py --seed 42 --tune-workers 20

# evaluate a checkpoint on the validation sites
CUDA_VISIBLE_DEVICES=0 python nave_evaluate.py runs/nave_s42_*/nave_best.pt --workers 13

# write the tuned per-class thresholds to JSON; fp16 inference for speed
CUDA_VISIBLE_DEVICES=0 python nave_evaluate.py runs/nave_s42_*/nave_best.pt \
    --workers 13 --fp16 --out thresholds.json
```

Each training run writes `runs/nave_s<seed>_<timestamp>/nave_best.pt` (best tuned
macro) and `nave_epoch_NN.pt`. Every checkpoint stores `model_state_dict`, the
tuned per-class `thresholds`, `macro_f1`, `epoch` and `seed`.

`nave_evaluate.py` runs one forward pass over `cfg.VAL_DATASETS`, tunes per-class
thresholds with the same coordinate-descent grid the training loop uses, and
prints the tuned thresholds plus the macro F1 (the official challenge metric,
labelled simply `F1` in the output). Pass `--fp16` for autocast inference and
`--out path.json` to dump the tuned thresholds.

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
`nave_train`. `nave_evaluate.py` is exactly that pattern in ~100 lines.

## Logging

Training progress -- per-epoch train/val loss, per-class P/R/F1, the tuned
per-class thresholds and the running best macro F1 -- is printed to stdout.
There is no external experiment tracker.
