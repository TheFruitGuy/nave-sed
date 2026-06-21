"""
NAVE training entry point (self-contained).
===========================================
Trains a single NAVE model and checkpoints the EMA weights selected by tuned
macro F1. Logs progress to stdout.

    CUDA_VISIBLE_DEVICES=0 python nave_train.py --seed 42
    CUDA_VISIBLE_DEVICES=1 python nave_train.py --seed 2024 --tune-workers 20

Checkpoints land in ``runs/nave_s<seed>_<timestamp>/`` as ``nave_best.pt`` (best
tuned macro F1) and ``nave_epoch_NN.pt``. They load with
``NAVE().load_checkpoint(...)``.

Self-contained
--------------
This module bundles the entire training + evaluation pipeline so the NAVE line
runs from only four files: ``nave_config`` / ``nave_features`` / ``nave_model``
/ ``nave_train``. The data loader, per-epoch negative resampling, post-
processing, event-level metrics, per-class threshold tuner, EMA, optimiser and
validation paths are all defined here. Nothing outside these four files (and
the third-party packages listed in the README) is imported.

Directory layout expected by the loader (set ``cfg.DATA_ROOT``)::

    DATA_ROOT/
      train/      annotations/{dataset}.csv   audio/{dataset}/*.wav
      validation/ annotations/{dataset}.csv   audio/{dataset}/*.wav
"""

from __future__ import annotations

import argparse
import hashlib
import math
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from scipy.ndimage import median_filter
from torch.utils.data import DataLoader, Dataset

import nave_config as cfg
from nave_model import NAVE
from nave_features import NAVEFeatureExtractor

try:                                   # tqdm is optional
    from tqdm import tqdm
except Exception:                      # pragma: no cover
    def tqdm(it, **kw):
        return it


# ======================================================================
# Reproducibility helpers
# ======================================================================

def seed_everything(seed: int = 42, deterministic: bool = False) -> int:
    """Seed Python, NumPy and PyTorch (CPU + CUDA). Call once before any model,
    dataset or DataLoader is built (workers inherit the RNG at construction)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def seeded_dataloader_kwargs(seed: int) -> dict:
    """DataLoader kwargs giving reproducible shuffle and worker RNG."""
    g = torch.Generator()
    g.manual_seed(seed)

    def _worker_init(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return {"generator": g, "worker_init_fn": _worker_init}


# ======================================================================
# Disk cache (parquet, keyed on the sorted dataset list)
# ======================================================================

_CACHE_DIR = Path("./.cache")
_CACHE_EXT = ".parquet"


def _cache_path(name: str, datasets: list[str]) -> Path:
    _CACHE_DIR.mkdir(exist_ok=True)
    key = hashlib.md5(",".join(sorted(datasets)).encode()).hexdigest()[:8]
    return _CACHE_DIR / f"{name}_{key}{_CACHE_EXT}"


def _cache_load(path: Path) -> "pd.DataFrame | None":
    """Load a cached DataFrame, or return None on miss / corruption / no engine."""
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"  Cache unavailable ({path.name}: {e}); rebuilding")
        return None


def _cache_save(path: Path, df: pd.DataFrame) -> None:
    """Atomically write a DataFrame to parquet; skip silently if no engine."""
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)
    except Exception as e:
        print(f"  Cache write skipped ({path.name}: {e}); continuing without cache")


def clear_cache() -> None:
    """Remove all cached manifests and annotations."""
    if _CACHE_DIR.exists():
        for f in _CACHE_DIR.glob(f"*{_CACHE_EXT}"):
            f.unlink()
        print(f"Cleared cache directory: {_CACHE_DIR}")


# ======================================================================
# Path resolution
# ======================================================================

def _split_for_dataset(ds: str) -> str:
    """Return the split directory ("train"/"validation") for a dataset name."""
    if ds in cfg.TRAIN_DATASETS:
        return "train"
    if ds in cfg.VAL_DATASETS:
        return "validation"
    for split in ("train", "validation"):
        if (cfg.DATA_ROOT / split / "audio" / ds).exists():
            return split
    raise FileNotFoundError(f"Cannot find split for dataset '{ds}'")


def _parse_file_start_dt(filename: str):
    """Parse the UTC start datetime encoded in an ATBFL audio filename, e.g.
    ``2014-06-29T23-00-00_000.wav`` -> 2014-06-29 23:00:00 UTC. None on mismatch."""
    stem = Path(filename).stem.split("_")[0]
    try:
        return datetime.strptime(stem, "%Y-%m-%dT%H-%M-%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# ======================================================================
# File manifest
# ======================================================================

def _build_file_manifest_uncached(datasets: list[str]) -> pd.DataFrame:
    """Scan the filesystem and read each WAV header for its duration."""
    rows = []
    for ds in datasets:
        try:
            split = _split_for_dataset(ds)
        except FileNotFoundError:
            print(f"Warning: cannot locate {ds}")
            continue

        audio_dir = cfg.DATA_ROOT / split / "audio" / ds
        if not audio_dir.exists():
            print(f"Warning: audio directory missing for {ds}: {audio_dir}")
            continue

        for wav in sorted(audio_dir.glob("*.wav")):
            info = sf.info(str(wav))
            start_dt = _parse_file_start_dt(wav.name)
            end_dt = start_dt + timedelta(seconds=info.duration) if start_dt else None
            rows.append({
                "dataset": ds,
                "filename": wav.name,
                "path": str(wav),
                "duration_s": info.duration,
                "start_dt": start_dt,
                "end_dt": end_dt,
            })

    return pd.DataFrame(rows)


def get_file_manifest(datasets: list[str]) -> pd.DataFrame:
    """DataFrame of all audio files in the requested datasets, cached to disk.
    Columns: dataset, filename, path, duration_s, start_dt, end_dt."""
    cp = _cache_path("manifest", datasets)
    cached = _cache_load(cp)
    if cached is not None:
        return cached

    print(f"  Building file manifest for {len(datasets)} dataset(s)...")
    df = _build_file_manifest_uncached(datasets)
    _cache_save(cp, df)
    return df


# ======================================================================
# Annotation loading
# ======================================================================

def _infer_filenames_vectorized(df: pd.DataFrame, ds_files: pd.DataFrame) -> pd.Series:
    """Infer the owning file for each annotation via ``pd.merge_asof`` (files in
    a dataset are non-overlapping in time, so the backward match is unique)."""
    if df.empty or ds_files.empty:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")

    df_sorted = df.sort_values("start_datetime").copy()
    files_sorted = (
        ds_files[["start_dt", "end_dt", "filename"]]
        .sort_values("start_dt")
        .reset_index(drop=True)
    )

    merged = pd.merge_asof(
        df_sorted, files_sorted,
        left_on="start_datetime", right_on="start_dt",
        direction="backward",
    )

    out_of_range = merged["start_datetime"] >= merged["end_dt"]
    merged.loc[out_of_range, "filename"] = pd.NA

    merged.index = df_sorted.index
    return merged.sort_index()["filename"]


def _load_annotations_uncached(datasets: list[str],
                               manifest: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """Read annotation CSVs and infer filenames where absent."""
    all_rows = []
    if manifest is None:
        manifest = get_file_manifest(datasets)

    for ds in datasets:
        try:
            split = _split_for_dataset(ds)
        except FileNotFoundError:
            continue
        ann_path = cfg.DATA_ROOT / split / "annotations" / f"{ds}.csv"
        if not ann_path.exists():
            print(f"Warning: no annotations for {ds}: {ann_path}")
            continue

        df = pd.read_csv(ann_path)
        df["dataset"] = ds
        df["start_datetime"] = pd.to_datetime(df["start_datetime"], utc=True)
        df["end_datetime"] = pd.to_datetime(df["end_datetime"], utc=True)

        if "filename" not in df.columns:
            ds_files = manifest[manifest["dataset"] == ds]
            df["filename"] = _infer_filenames_vectorized(df, ds_files)

            n_before = len(df)
            df = df[df["filename"].notna()].reset_index(drop=True)
            n_dropped = n_before - len(df)
            if n_dropped > 0:
                print(f"  {ds}: dropped {n_dropped}/{n_before} annotations "
                      f"with no matching file")

        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    ann = pd.concat(all_rows, ignore_index=True)
    ann["label_3class"] = ann["annotation"].map(cfg.COLLAPSE_MAP).fillna(ann["annotation"])
    return ann


def load_annotations(datasets: list[str],
                     manifest: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """Load and concatenate annotations for the requested datasets, cached to disk.
    Columns: dataset, filename, start_datetime, end_datetime, annotation (fine
    7-class), label_3class (coarse)."""
    cp = _cache_path("annotations", datasets)
    cached = _cache_load(cp)
    if cached is not None:
        return cached

    print(f"  Loading annotations for {len(datasets)} dataset(s)...")
    df = _load_annotations_uncached(datasets, manifest=manifest)
    _cache_save(cp, df)
    return df


# ======================================================================
# Segment dataclass and grouping
# ======================================================================

@dataclass
class Segment:
    """Lightweight description of an audio segment, loaded on demand."""

    dataset: str
    filename: str
    path: str
    start_sample: int
    end_sample: int
    file_start_dt: datetime
    annotations: list[dict]
    is_positive: bool


def _build_annotations_by_file(annotations: pd.DataFrame, manifest: pd.DataFrame) -> dict:
    """Group annotations by ``(dataset, filename)`` with file-relative offsets."""
    if annotations.empty or manifest.empty:
        return {}

    file_starts = {
        (r["dataset"], r["filename"]): r["start_dt"]
        for _, r in manifest.iterrows()
    }

    out: dict = {}
    for _, a in annotations.iterrows():
        key = (a["dataset"], a["filename"])
        fsd = file_starts.get(key)
        if fsd is None or pd.isna(fsd):
            continue
        out.setdefault(key, []).append({
            "start_s": (a["start_datetime"] - fsd).total_seconds(),
            "end_s": (a["end_datetime"] - fsd).total_seconds(),
            "label": a["annotation"],
            "label_3class": a["label_3class"],
        })
    return out


# ======================================================================
# Training segment construction
# ======================================================================

def build_positive_segments(annotations: pd.DataFrame, manifest: pd.DataFrame,
                            collar_min_s: float = cfg.COLLAR_MIN_S,
                            collar_max_s: float = cfg.COLLAR_MAX_S,
                            rng: "random.Random | None" = None) -> list[Segment]:
    """One training segment per valid positive annotation (call + random collar);
    all intersecting annotations are attached so multi-call windows label right."""
    if rng is None:
        rng = random

    segments: list[Segment] = []
    if manifest.empty or annotations.empty:
        return segments

    manifest_idx = manifest.set_index(["dataset", "filename"])
    ann_by_file = _build_annotations_by_file(annotations, manifest)

    for _, row in annotations.iterrows():
        key = (row["dataset"], row["filename"])
        if key not in manifest_idx.index:
            continue
        file_row = manifest_idx.loc[key]
        file_start_dt = file_row["start_dt"]
        if file_start_dt is None or pd.isna(file_start_dt):
            continue

        call_start_s = (row["start_datetime"] - file_start_dt).total_seconds()
        call_end_s = (row["end_datetime"] - file_start_dt).total_seconds()

        if call_end_s <= call_start_s or call_end_s <= 0:
            continue
        if call_end_s - call_start_s > cfg.MAX_CALL_DURATION_S:
            continue
        if call_end_s - call_start_s < cfg.MIN_CALL_DURATION_S:
            continue

        pre = rng.uniform(collar_min_s, collar_max_s)
        post = rng.uniform(collar_min_s, collar_max_s)
        seg_start_s = max(0.0, call_start_s - pre)
        seg_end_s = min(file_row["duration_s"], call_end_s + post)

        file_anns = ann_by_file.get(key, [])
        inter_anns = [
            a for a in file_anns
            if a["end_s"] > seg_start_s and a["start_s"] < seg_end_s
        ]

        segments.append(Segment(
            dataset=row["dataset"],
            filename=row["filename"],
            path=file_row["path"],
            start_sample=int(seg_start_s * cfg.SAMPLE_RATE),
            end_sample=int(seg_end_s * cfg.SAMPLE_RATE),
            file_start_dt=file_start_dt,
            annotations=inter_anns,
            is_positive=True,
        ))

    return segments


def build_negative_segments(annotations: pd.DataFrame, manifest: pd.DataFrame,
                            n_segments: int, min_dur_s: float = 5.0,
                            max_dur_s: float = 30.0,
                            rng: "random.Random | None" = None) -> list[Segment]:
    """Sample up to ``n_segments`` random call-free windows (rejection sampling
    against annotated calls; retry cap 20x ``n_segments``)."""
    if rng is None:
        rng = random

    segments: list[Segment] = []
    if manifest.empty:
        return segments

    ann_by_file = _build_annotations_by_file(annotations, manifest)
    call_intervals: dict = {
        key: [(a["start_s"], a["end_s"]) for a in anns]
        for key, anns in ann_by_file.items()
    }

    files = manifest.to_dict("records")
    tries, max_tries = 0, n_segments * 20

    while len(segments) < n_segments and tries < max_tries:
        tries += 1
        file_row = rng.choice(files)
        key = (file_row["dataset"], file_row["filename"])
        dur = file_row["duration_s"]
        seg_len = rng.uniform(min_dur_s, max_dur_s)

        if dur <= seg_len + 1.0:
            continue

        seg_start_s = rng.uniform(0, dur - seg_len)
        seg_end_s = seg_start_s + seg_len

        intervals = call_intervals.get(key, [])
        overlap = any(seg_end_s > cs and seg_start_s < ce for cs, ce in intervals)
        if overlap:
            continue

        segments.append(Segment(
            dataset=file_row["dataset"],
            filename=file_row["filename"],
            path=file_row["path"],
            start_sample=int(seg_start_s * cfg.SAMPLE_RATE),
            end_sample=int(seg_end_s * cfg.SAMPLE_RATE),
            file_start_dt=file_row["start_dt"],
            annotations=[],
            is_positive=False,
        ))

    return segments


# ======================================================================
# Fixed-length segment extension
# ======================================================================

def extend_segment_to_fixed_length(seg: Segment, target_seconds: float,
                                   file_duration_s: float,
                                   sample_rate: int = cfg.SAMPLE_RATE,
                                   rng: random.Random = None) -> Segment:
    """Return a copy of ``seg`` whose window is exactly ``target_seconds`` long,
    with the original content's position within the new window randomised."""
    if rng is None:
        rng = random

    target_samples = int(target_seconds * sample_rate)
    file_samples = int(file_duration_s * sample_rate)
    cur_length = seg.end_sample - seg.start_sample

    if cur_length >= target_samples:
        return seg

    extra = target_samples - cur_length

    if file_samples <= target_samples:
        return replace(seg, start_sample=0, end_sample=file_samples)

    pre_room = seg.start_sample
    post_room = file_samples - seg.end_sample

    pre_extra = min(pre_room, rng.randint(0, extra))
    post_extra = min(post_room, extra - pre_extra)

    deficit = extra - pre_extra - post_extra
    if deficit > 0:
        if pre_room - pre_extra >= deficit:
            pre_extra += deficit
        else:
            post_extra += deficit

    new_start = max(0, seg.start_sample - pre_extra)
    new_end = min(file_samples, new_start + target_samples)
    new_start = max(0, new_end - target_samples)

    return replace(seg, start_sample=new_start, end_sample=new_end)


def extend_all_segments(segments, manifest, target_seconds: float):
    """Apply :func:`extend_segment_to_fixed_length` to every segment (fixed-seed
    RNG so the randomised positioning is reproducible)."""
    rng = random.Random(0xC0FFEE)
    duration_lookup = {
        (r["dataset"], r["filename"]): r["duration_s"]
        for _, r in manifest.iterrows()
    }
    extended = []
    for seg in segments:
        dur = duration_lookup.get((seg.dataset, seg.filename))
        if dur is None:
            continue
        extended.append(
            extend_segment_to_fixed_length(seg, target_seconds, dur, rng=rng)
        )
    return extended


# ======================================================================
# Validation segment construction
# ======================================================================

def build_val_segments(manifest: pd.DataFrame, annotations: pd.DataFrame,
                       segment_s: float = cfg.EVAL_SEGMENT_S,
                       overlap_s: float = cfg.EVAL_OVERLAP_S) -> list[Segment]:
    """Tile each file with fixed-length overlapping windows for evaluation."""
    segments: list[Segment] = []
    if manifest.empty:
        return segments

    step_s = segment_s - overlap_s
    ann_by_file = _build_annotations_by_file(annotations, manifest)

    for _, f in manifest.iterrows():
        key = (f["dataset"], f["filename"])
        dur = f["duration_s"]
        fsd = f["start_dt"]
        file_anns = ann_by_file.get(key, [])

        t = 0.0
        while t + segment_s <= dur + 1e-6:
            inter = [a for a in file_anns
                     if a["end_s"] > t and a["start_s"] < t + segment_s]
            segments.append(Segment(
                dataset=f["dataset"],
                filename=f["filename"],
                path=f["path"],
                start_sample=int(t * cfg.SAMPLE_RATE),
                end_sample=int((t + segment_s) * cfg.SAMPLE_RATE),
                file_start_dt=fsd,
                annotations=inter,
                is_positive=len(inter) > 0,
            ))
            t += step_s

    return segments


# ======================================================================
# PyTorch Dataset and collation
# ======================================================================

class WhaleDataset(Dataset):
    """Map-style dataset yielding ``(audio, targets, mask, meta)`` per segment.
    Audio is read lazily from disk; the target width follows ``cfg.n_classes()``."""

    def __init__(self, segments: list[Segment]):
        self.segments = segments
        self.stride_samp = int(cfg.FRAME_STRIDE_S * cfg.SAMPLE_RATE)
        self.class_idx = cfg.class_to_idx()
        self.n_classes = cfg.n_classes()

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int):
        seg = self.segments[idx]
        n_samples = seg.end_sample - seg.start_sample

        audio, sr = sf.read(
            seg.path, start=seg.start_sample, stop=seg.end_sample, dtype="float32"
        )
        assert sr == cfg.SAMPLE_RATE, f"Expected {cfg.SAMPLE_RATE} Hz, got {sr}"
        audio = torch.from_numpy(audio)

        n_frames = n_samples // self.stride_samp
        targets = torch.zeros(n_frames, self.n_classes)
        seg_start_s = seg.start_sample / cfg.SAMPLE_RATE

        for a in seg.annotations:
            label = a["label_3class"] if cfg.USE_3CLASS else a["label"]
            if label not in self.class_idx:
                continue
            c = self.class_idx[label]
            local_start_s = max(0.0, a["start_s"] - seg_start_s)
            local_end_s = min(n_samples / cfg.SAMPLE_RATE, a["end_s"] - seg_start_s)
            f0 = int(local_start_s / cfg.FRAME_STRIDE_S)
            f1 = int(local_end_s / cfg.FRAME_STRIDE_S)
            targets[f0:f1, c] = 1.0

        mask = torch.ones(n_frames, dtype=torch.bool)
        meta = {
            "dataset": seg.dataset,
            "filename": seg.filename,
            "start_sample": seg.start_sample,
            "end_sample": seg.end_sample,
        }
        return audio, targets, mask, meta


def collate_fn(batch):
    """Pad variable-length segments to the batch maximum."""
    audios, targets, masks, metas = zip(*batch)
    max_samp = max(a.size(0) for a in audios)
    max_frames = max(t.size(0) for t in targets)
    n_classes = targets[0].size(1)
    B = len(audios)

    audio_pad = torch.zeros(B, max_samp)
    target_pad = torch.zeros(B, max_frames, n_classes)
    mask_pad = torch.zeros(B, max_frames, dtype=torch.bool)

    for i in range(B):
        audio_pad[i, :audios[i].size(0)] = audios[i]
        target_pad[i, :targets[i].size(0)] = targets[i]
        mask_pad[i, :masks[i].size(0)] = masks[i]

    return audio_pad, target_pad, mask_pad, list(metas)


# ======================================================================
# Per-epoch negative resampling
# ======================================================================

def resample_negatives_for_epoch(pos_segs_extended: list, train_annotations,
                                 train_manifest, n_neg: int, segment_s: float,
                                 epoch: int, verbose: bool = False):
    """Draw a fresh negative segment set for one epoch and return the combined
    (fixed positives + new negatives) training segment list. No per-epoch seed
    is derived: the global RNG advances naturally so each epoch differs while the
    whole run stays reproducible from the master seed."""
    neg_segs = build_negative_segments(
        train_annotations, train_manifest, n_segments=n_neg,
    )
    neg_segs = extend_all_segments(neg_segs, train_manifest, segment_s)

    if verbose and neg_segs:
        first = neg_segs[0]
        print(f"    epoch {epoch}: resampled {len(neg_segs)} negatives "
              f"[first: {first.filename} @ {first.start_sample} samp]")

    return pos_segs_extended + neg_segs


# ======================================================================
# Class weights
# ======================================================================

def compute_pos_weight(sites: list[str], device: torch.device, verbose: bool = True,
                       annotations=None) -> "tuple[torch.Tensor, dict]":
    """Per-class ``pos_weight`` (``w_c = N / P_c``) for the active head, counted
    over the raw annotation table before any dataloader exists."""
    if annotations is None:
        annotations = load_annotations(sites)

    class_labels = cfg.class_names()
    label_col = "label_3class" if cfg.USE_3CLASS else "annotation"
    p_c = [max(int((annotations[label_col] == c).sum()), 1)
           for c in class_labels]
    n_total = sum(p_c)
    weights = [n_total / pc for pc in p_c]

    info = {
        "annotation_counts": dict(zip(class_labels, p_c)),
        "pos_weight": dict(zip(class_labels, weights)),
        "n_total_pos": n_total,
        "min_weight": min(weights),
        "max_weight": max(weights),
        "weight_ratio": max(weights) / min(weights),
    }

    if verbose:
        print(f"\n  Per-class positive weights (w_c = N / P_c) "
              f"[{len(class_labels)}-class head]:")
        print(f"  {'class':12} {'P_c (segments)':>15} {'w_c':>10}")
        for c_name, pc, w in zip(class_labels, p_c, weights):
            print(f"  {c_name:12} {pc:>15,} {w:>10.3f}")
        print(f"  {'total':12} {n_total:>15,}")
        print(f"  Weight ratio (max/min): {info['weight_ratio']:.2f}x")

    return torch.tensor(weights, dtype=torch.float32).to(device), info


# ======================================================================
# Post-processing: 7->3 collapse, stitch, smooth, threshold, merge
# ======================================================================

_SEVEN_TO_THREE = {
    "bmabz": [cfg.CALL_TYPES_7.index(x) for x in ("bma", "bmb", "bmz")],
    "d":     [cfg.CALL_TYPES_7.index(x) for x in ("bmd", "bpd")],
    "bp":    [cfg.CALL_TYPES_7.index(x) for x in ("bp20", "bp20plus")],
}


def collapse_probs_to_3class(all_probs: dict) -> dict:
    """Collapse per-window 7-class probabilities to 3-class via max-pooling within
    each coarse group. Returns the input unchanged when ``cfg.USE_3CLASS`` is True
    or the arrays are not 7-wide (so it is safe to call unconditionally)."""
    if cfg.USE_3CLASS or not all_probs:
        return all_probs

    sample = next(iter(all_probs.values()))
    if sample.shape[1] != 7:
        return all_probs

    out = {}
    for key, p7 in all_probs.items():
        p3 = np.zeros((p7.shape[0], 3), dtype=p7.dtype)
        for i, name in enumerate(cfg.CALL_TYPES_3):
            p3[:, i] = p7[:, _SEVEN_TO_THREE[name]].max(axis=1)
        out[key] = p3
    return out


@dataclass
class Detection:
    """A single predicted or ground-truth event (file-relative seconds)."""

    dataset: str
    filename: str
    label: str
    start_s: float
    end_s: float
    confidence: float = 1.0


def stitch_segments(all_probs: "dict") -> dict:
    """Merge overlapping-window predictions into one per-file probability stream
    by averaging predictions in overlap regions."""
    stride_samp = int(cfg.FRAME_STRIDE_S * cfg.SAMPLE_RATE)

    file_segs: dict = {}
    for (ds, fn, start_samp), probs in all_probs.items():
        file_segs.setdefault((ds, fn), []).append((start_samp, probs))

    result = {}
    for key, segs in file_segs.items():
        segs.sort(key=lambda x: x[0])

        max_end = max(s + p.shape[0] * stride_samp for s, p in segs)
        total_frames = max_end // stride_samp + 1
        nc = segs[0][1].shape[1]

        accum = np.zeros((total_frames, nc), dtype=np.float64)
        counts = np.zeros(total_frames, dtype=np.float64)

        for start_samp, probs in segs:
            f0 = start_samp // stride_samp
            T = min(probs.shape[0], total_frames - f0)
            accum[f0:f0 + T] += probs[:T]
            counts[f0:f0 + T] += 1

        counts = np.maximum(counts, 1)
        result[key] = (accum / counts[:, None]).astype(np.float32)

    return result


def smooth_probabilities(probs: np.ndarray,
                         kernel_ms: int = cfg.SMOOTH_KERNEL_MS) -> np.ndarray:
    """Temporal median filter on per-frame class probabilities."""
    stride_ms = int(cfg.FRAME_STRIDE_S * 1000)
    k = max(1, kernel_ms // stride_ms)
    if k % 2 == 0:
        k += 1

    out = np.zeros_like(probs)
    for c in range(probs.shape[1]):
        out[:, c] = median_filter(probs[:, c], size=k)
    return out


def threshold_to_detections(probs: np.ndarray, thresholds: np.ndarray,
                            dataset: str, filename: str,
                            offset_sample: int = 0) -> list[Detection]:
    """One Detection per contiguous above-threshold run, per class."""
    names = cfg.class_names()
    dets = []
    T, C = probs.shape
    offset_s = offset_sample / cfg.SAMPLE_RATE

    for c in range(C):
        active = probs[:, c] > thresholds[c]
        diffs = np.diff(active.astype(int), prepend=0, append=0)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        for s, e in zip(starts, ends):
            dets.append(Detection(
                dataset=dataset,
                filename=filename,
                label=names[c],
                start_s=s * cfg.FRAME_STRIDE_S + offset_s,
                end_s=e * cfg.FRAME_STRIDE_S + offset_s,
                confidence=float(probs[s:e, c].mean()),
            ))

    return dets


def merge_and_filter(detections: list[Detection]) -> list[Detection]:
    """Collapse labels, merge nearby same-class detections (< ``MERGE_GAP_S``),
    and drop events outside ``[POST_MIN_DUR_S, POST_MAX_DUR_S]``."""
    collapsed = []
    for d in detections:
        new_label = cfg.COLLAPSE_MAP.get(d.label, d.label)
        collapsed.append(Detection(
            dataset=d.dataset, filename=d.filename, label=new_label,
            start_s=d.start_s, end_s=d.end_s, confidence=d.confidence,
        ))

    groups: dict = {}
    for d in collapsed:
        groups.setdefault((d.dataset, d.filename, d.label), []).append(d)

    final = []
    for _, events in groups.items():
        events.sort(key=lambda x: x.start_s)

        merged = []
        for e in events:
            if not merged:
                merged.append(e)
            else:
                last = merged[-1]
                if e.start_s - last.end_s <= cfg.MERGE_GAP_S:
                    last.end_s = max(last.end_s, e.end_s)
                    last.confidence = max(last.confidence, e.confidence)
                else:
                    merged.append(e)

        for m in merged:
            dur = m.end_s - m.start_s
            if cfg.POST_MIN_DUR_S <= dur <= cfg.POST_MAX_DUR_S:
                final.append(m)

    return final


def postprocess_predictions(all_probs: "dict",
                            thresholds: np.ndarray) -> list[Detection]:
    """Run stitch -> smooth -> threshold -> merge/filter end-to-end."""
    file_probs = stitch_segments(all_probs)
    all_dets = []
    for (ds, fn), probs in file_probs.items():
        probs = smooth_probabilities(probs)
        all_dets.extend(threshold_to_detections(probs, thresholds, ds, fn))
    return merge_and_filter(all_dets)


# ======================================================================
# Event-level evaluation
# ======================================================================

def compute_iou_1d(ps: float, pe: float, gs: float, ge: float) -> float:
    """1D IoU between intervals ``[ps, pe)`` and ``[gs, ge)``."""
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def compute_metrics(predictions, ground_truth, iou_threshold: float = 0.3) -> dict:
    """Per-class and overall precision / recall / F1 via greedy 1D IoU matching."""
    classes = sorted({d.label for d in list(predictions) + list(ground_truth)})
    results = {}
    tp_tot = fp_tot = fn_tot = 0

    for cls in classes:
        cp = [d for d in predictions if d.label == cls]
        cg = [d for d in ground_truth if d.label == cls]
        files = {(d.dataset, d.filename) for d in cp + cg}
        tp = fp = fn = 0

        for fk in files:
            file_preds = sorted([d for d in cp if (d.dataset, d.filename) == fk],
                                key=lambda x: x.start_s)
            file_gts = sorted([d for d in cg if (d.dataset, d.filename) == fk],
                              key=lambda x: x.start_s)
            matched = set()

            for gt in file_gts:
                best_iou, best_i = 0.0, -1
                for i, pr in enumerate(file_preds):
                    if i in matched:
                        continue
                    iou = compute_iou_1d(pr.start_s, pr.end_s, gt.start_s, gt.end_s)
                    if iou > best_iou:
                        best_iou, best_i = iou, i
                if best_iou >= iou_threshold and best_i >= 0:
                    tp += 1
                    matched.add(best_i)
                else:
                    fn += 1

            fp += len(file_preds) - len(matched)

        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        results[cls] = {
            "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r + 1e-8),
            "tp": tp, "fp": fp, "fn": fn,
        }
        tp_tot += tp
        fp_tot += fp
        fn_tot += fn

    p = tp_tot / (tp_tot + fp_tot + 1e-8)
    r = tp_tot / (tp_tot + fn_tot + 1e-8)
    results["overall"] = {"precision": p, "recall": r,
                          "f1": 2 * p * r / (p + r + 1e-8)}
    return results


# ======================================================================
# Per-class threshold tuner (coordinate descent; fork-parallel within a class)
# ======================================================================

CLASS_NAMES: list[str] = list(cfg.CALL_TYPES_3)
BMABZ_GRID = np.arange(0.20, 0.85, 0.05)
D_GRID = np.concatenate([np.arange(0.05, 0.5, 0.05), np.arange(0.5, 0.85, 0.10)])
BP_GRID = np.concatenate([np.arange(0.05, 0.5, 0.05), np.arange(0.5, 0.85, 0.10)])
THRESHOLD_GRIDS = [BMABZ_GRID, D_GRID, BP_GRID]
IOU_THRESHOLD = cfg.IOU_THRESHOLD

_WORKER_PROBS: "dict | None" = None
_WORKER_GT: "list | None" = None


def _grid_task(payload):
    c, t_val, trial = payload
    preds = postprocess_predictions(_WORKER_PROBS, trial)
    metrics = compute_metrics(preds, _WORKER_GT, iou_threshold=IOU_THRESHOLD)
    return c, float(t_val), float(metrics.get(CLASS_NAMES[c], {}).get("f1", 0.0))


def tune_thresholds_per_class(probs, gt_events, start=None, workers=1):
    """Per-class coordinate descent (BMABZ -> D -> BP). Parallel within a class."""
    thr = (np.array([0.5, 0.5, 0.5], dtype=np.float64) if start is None
           else np.asarray(start, dtype=np.float64).copy())

    if workers <= 1:
        for c, name in enumerate(CLASS_NAMES):
            best_f1, best_t = -1.0, thr[c]
            for t in THRESHOLD_GRIDS[c]:
                trial = thr.copy(); trial[c] = float(t)
                m = compute_metrics(postprocess_predictions(probs, trial),
                                    gt_events, iou_threshold=IOU_THRESHOLD)
                f1 = m.get(name, {}).get("f1", 0.0)
                if f1 > best_f1:
                    best_f1, best_t = f1, float(t)
            thr[c] = best_t
        return thr

    ctx = mp.get_context("fork")
    global _WORKER_PROBS, _WORKER_GT
    _WORKER_PROBS, _WORKER_GT = probs, list(gt_events)
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            for c, name in enumerate(CLASS_NAMES):
                tasks = []
                for t in THRESHOLD_GRIDS[c]:
                    trial = thr.copy(); trial[c] = float(t)
                    tasks.append((c, float(t), trial))
                best_f1, best_t = -1.0, thr[c]
                for fut in as_completed([pool.submit(_grid_task, tk) for tk in tasks]):
                    _, t_val, f1 = fut.result()
                    if f1 > best_f1:
                        best_f1, best_t = f1, t_val
                thr[c] = best_t
    finally:
        _WORKER_PROBS = _WORKER_GT = None
    return thr


# ======================================================================
# EMA of weights (eval-time teacher)
# ======================================================================

class EMA:
    """EMA over the model's float tensors (params + BN running stats)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }
        self._backup: dict = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def store_and_copy(self, model: nn.Module):
        """Stash current weights and load the EMA weights for evaluation."""
        msd = model.state_dict()
        self._backup = {k: msd[k].detach().clone() for k in self.shadow}
        for k in self.shadow:
            msd[k].copy_(self.shadow[k])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        """Put the stashed training weights back."""
        msd = model.state_dict()
        for k, v in self._backup.items():
            msd[k].copy_(v)
        self._backup = {}


# ======================================================================
# Optimiser + warmup/decay scheduler
# ======================================================================

def build_optim_sched(model: nn.Module, opt_kwargs: dict,
                      steps_per_epoch: int, total_epochs: int):
    """Build ``(optimizer, scheduler, is_schedule_free)`` from ``opt_kwargs``:
    optimizer ("radam"|"adamw"), peak_lr, weight_decay, warmup_epochs (linear
    warmup 0->peak), schedule ("cosine"|"step"|"const"). The scheduler is a
    per-step LambdaLR; ``is_schedule_free`` is always False here. The defaults
    in ``nave_config`` use plain RAdam at constant LR (warmup 0, schedule "const")."""
    name = opt_kwargs.get("optimizer", "radam").lower()
    peak_lr = float(opt_kwargs.get("peak_lr", cfg.LR))
    wd = float(opt_kwargs.get("weight_decay", cfg.WEIGHT_DECAY))
    betas = (cfg.BETA1, cfg.BETA2)

    if name in ("sf-radam", "radam-sf", "schedulefree-radam"):
        raise ValueError(
            "schedule-free RAdam is not bundled in the standalone NAVE pipeline; "
            "use cfg.OPTIMIZER 'radam' (the default) or 'adamw'.")

    if name == "radam":
        optimizer = torch.optim.RAdam(model.parameters(), lr=peak_lr,
                                      weight_decay=wd, betas=betas)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr,
                                      weight_decay=wd, betas=betas)

    warmup_epochs = float(opt_kwargs.get("warmup_epochs", 0.0))
    schedule = opt_kwargs.get("schedule", "const").lower()
    total_steps = max(1, steps_per_epoch * total_epochs)
    warmup_steps = int(steps_per_epoch * warmup_epochs)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule == "cosine":
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        if schedule == "step":
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.1 ** int(min(0.999, progress) * 3)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler, False


# ======================================================================
# Loss
# ======================================================================

class MaskedBCELoss(nn.Module):
    """Masked, segment-count-weighted BCE: per_frame = BCE(none)*valid, then
    sum / (valid * C)."""

    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    def forward(self, logits, targets, valid):            # valid (B,T)
        v = valid.unsqueeze(-1).float()
        per = self.bce(logits, targets) * v
        return per.sum() / (v.sum() * targets.size(-1)).clamp(min=1.0)


# ======================================================================
# Train / validate
# ======================================================================

def _train_epoch(model, spec_extractor, loader, criterion, optimizer,
                 scheduler, device, ema=None):
    model.train()
    losses, n = 0.0, 0
    for audio, targets, mask, _ in tqdm(loader, desc="Train", leave=False):
        audio = audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        spec = spec_extractor(audio)
        logits = model(spec, key_padding_mask=~mask.bool())
        T = min(logits.size(1), targets.size(1))
        logits, targets_t, mask_t = logits[:, :T], targets[:, :T], mask[:, :T]

        loss = criterion(logits, targets_t, mask_t)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

        losses += loss.item()
        n += 1
    return losses / max(n, 1)


def _macro_paper(per_class) -> float:
    """Paper-convention macro F1: F1 of the mean P and mean R across the 3 classes."""
    p_bar = sum(per_class[n]["precision"] for n in cfg.CALL_TYPES_3) / 3
    r_bar = sum(per_class[n]["recall"] for n in cfg.CALL_TYPES_3) / 3
    return 2.0 * p_bar * r_bar / (p_bar + r_bar + 1e-8)


def _per_class_from_metrics(metrics) -> dict:
    return {
        name: {
            "f1": metrics.get(name, {}).get("f1", 0.0),
            "precision": metrics.get(name, {}).get("precision", 0.0),
            "recall": metrics.get(name, {}).get("recall", 0.0),
            "tp": metrics.get(name, {}).get("tp", 0),
            "fp": metrics.get(name, {}).get("fp", 0),
            "fn": metrics.get(name, {}).get("fn", 0),
        }
        for name in cfg.CALL_TYPES_3
    }


@torch.no_grad()
def validate(model, spec_extractor, loader, criterion, device,
             val_annotations, file_start_dts, threshold: float):
    """Fixed-threshold validation: collect probabilities, post-process at a single
    threshold, score event-level F1 against the 3-class ground truth."""
    model.eval()
    losses, n = 0.0, 0
    all_probs = {}
    hop = spec_extractor.hop_length

    for audio, targets, mask, metas in tqdm(loader, desc="Val", leave=False):
        audio = audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        logits = model(spec_extractor(audio))
        T = min(logits.size(1), targets.size(1))
        logits, targets, mask = logits[:, :T], targets[:, :T], mask[:, :T]

        valid = mask.unsqueeze(-1).float()
        per_frame = criterion(logits, targets) * valid
        loss = per_frame.sum() / (valid.sum() * targets.size(-1)).clamp(min=1.0)
        losses += loss.item()
        n += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            all_probs[key] = probs[j, :n_frames, :]

    if cfg.USE_3CLASS:
        all_probs_3 = all_probs
        thresholds = np.array([threshold] * 3)
        pred_events = postprocess_predictions(all_probs_3, thresholds)
    else:
        all_probs_3 = collapse_probs_to_3class(all_probs)
        cfg.USE_3CLASS = True
        try:
            thresholds = np.array([threshold] * 3)
            pred_events = postprocess_predictions(all_probs_3, thresholds)
        finally:
            cfg.USE_3CLASS = False

    gt_events = []
    for _, row in val_annotations.iterrows():
        fsd = file_start_dts.get((row["dataset"], row["filename"]))
        if fsd is None:
            continue
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"],
            label=row["label_3class"],
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))

    metrics = compute_metrics(pred_events, gt_events, iou_threshold=cfg.IOU_THRESHOLD)
    per_class = _per_class_from_metrics(metrics)
    return {
        "loss": losses / max(n, 1),
        "f1": metrics.get("overall", {}).get("f1", 0.0),
        "macro_paper": _macro_paper(per_class),
        "per_class": per_class,
    }


@torch.no_grad()
def validate_tuned(model, spec_extractor, loader, criterion, device,
                   val_annotations, file_start_dts, *,
                   workers: int = 1, start_thr=None):
    """Single inference pass + per-class threshold tuning. Same return keys as
    :func:`validate`, plus ``thresholds``."""
    hop = spec_extractor.hop_length

    model.eval()
    losses, n = 0.0, 0
    all_probs = {}
    for audio, targets, mask, metas in tqdm(loader, desc="ValTuned", leave=False):
        audio = audio.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        logits = model(spec_extractor(audio))
        T = min(logits.size(1), targets.size(1))
        logits, targets, mask = logits[:, :T], targets[:, :T], mask[:, :T]
        valid = mask.unsqueeze(-1).float()
        per_frame = criterion(logits, targets) * valid
        loss = per_frame.sum() / (valid.sum() * targets.size(-1)).clamp(min=1.0)
        losses += loss.item()
        n += 1

        probs = torch.sigmoid(logits).float().cpu().numpy()
        for j, meta in enumerate(metas):
            key = (meta["dataset"], meta["filename"], meta["start_sample"])
            n_samp = meta["end_sample"] - meta["start_sample"]
            n_frames = min(n_samp // hop, probs[j].shape[0])
            all_probs[key] = probs[j, :n_frames, :]

    flipped = False
    if cfg.USE_3CLASS:
        all_probs_3 = all_probs
    else:
        all_probs_3 = collapse_probs_to_3class(all_probs)
        cfg.USE_3CLASS = True
        flipped = True

    gt_events = []
    for _, row in val_annotations.iterrows():
        fsd = file_start_dts.get((row["dataset"], row["filename"]))
        if fsd is None:
            continue
        gt_events.append(Detection(
            dataset=row["dataset"], filename=row["filename"],
            label=row["label_3class"],
            start_s=(row["start_datetime"] - fsd).total_seconds(),
            end_s=(row["end_datetime"] - fsd).total_seconds(),
        ))

    try:
        thr = tune_thresholds_per_class(all_probs_3, gt_events,
                                        start=start_thr, workers=workers)
        metrics = compute_metrics(
            postprocess_predictions(all_probs_3, np.asarray(thr, dtype=np.float64)),
            gt_events, iou_threshold=cfg.IOU_THRESHOLD,
        )
    finally:
        if flipped:
            cfg.USE_3CLASS = False

    per_class = _per_class_from_metrics(metrics)
    return {
        "loss": losses / max(n, 1),
        "f1": metrics.get("overall", {}).get("f1", 0.0),
        "macro_paper": _macro_paper(per_class),
        "per_class": per_class,
        "thresholds": [float(t) for t in np.asarray(thr).ravel()],
    }


# ======================================================================
# CLI / main
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=cfg.SEED,
                   help="Run seed (vary it to build the ensemble).")
    p.add_argument("--tune-workers", type=int, default=8,
                   help="Parallel workers for the per-epoch threshold tuner.")
    return p.parse_args()


def main():
    args = parse_args()
    seed = seed_everything(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = cfg.EPOCHS
    print(f"[NAVE] device={device}  seed={seed}  epochs={epochs}  k={cfg.CONV_KERNEL}")

    train_sites, val_sites = list(cfg.TRAIN_DATASETS), list(cfg.VAL_DATASETS)
    pos_weight, _ = compute_pos_weight(train_sites, device, verbose=True)

    run_dir = Path(cfg.OUTPUT_DIR) / f"nave_s{seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[NAVE] run dir: {run_dir}")

    # --- data ---
    train_manifest = get_file_manifest(train_sites)
    train_annotations = load_annotations(train_sites, manifest=train_manifest)
    pos_segs = build_positive_segments(train_annotations, train_manifest)
    pos_segs = extend_all_segments(pos_segs, train_manifest, cfg.TRAIN_SEGMENT_S)
    n_neg = int(len(pos_segs) * cfg.NEG_RATIO)

    val_manifest = get_file_manifest(val_sites)
    val_annotations = load_annotations(val_sites, manifest=val_manifest)
    val_segments = build_val_segments(val_manifest, val_annotations)
    val_loader = DataLoader(
        WhaleDataset(val_segments), batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    file_start_dts = {(r["dataset"], r["filename"]): r["start_dt"]
                      for _, r in val_manifest.iterrows()}

    # --- model / loss / optim ---
    model = NAVE().to(device)
    spec_extractor = NAVEFeatureExtractor().to(device)
    print(f"[NAVE] parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_criterion = MaskedBCELoss(pos_weight=pos_weight).to(device)
    val_criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight).to(device)

    opt_kwargs = dict(optimizer=cfg.OPTIMIZER, peak_lr=cfg.LR, warmup_epochs=0.0,
                      schedule="const", weight_decay=cfg.WEIGHT_DECAY)
    steps_per_epoch = max(1, (len(pos_segs) + n_neg) // cfg.BATCH_SIZE)
    optimizer, scheduler, is_sf = build_optim_sched(model, opt_kwargs, steps_per_epoch, epochs)
    ema = EMA(model, decay=cfg.EMA_DECAY)

    # --- loop ---
    best_macro, last_thr = 0.0, None
    train_loader = None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        if train_loader is None or (epoch - 1) % cfg.RESAMPLE_EVERY == 0:
            train_segments = resample_negatives_for_epoch(
                pos_segs_extended=pos_segs, train_annotations=train_annotations,
                train_manifest=train_manifest, n_neg=n_neg,
                segment_s=cfg.TRAIN_SEGMENT_S, epoch=epoch, verbose=True)
            train_loader = DataLoader(
                WhaleDataset(train_segments), batch_size=cfg.BATCH_SIZE, shuffle=True,
                num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
                **seeded_dataloader_kwargs(seed))

        train_loss = _train_epoch(model, spec_extractor, train_loader, train_criterion,
                                  optimizer, scheduler, device, ema)

        # select / checkpoint on the EMA weights
        ema.store_and_copy(model)
        try:
            val = validate_tuned(model, spec_extractor, val_loader, val_criterion, device,
                                 val_annotations, file_start_dts,
                                 workers=args.tune_workers, start_thr=last_thr)
            last_thr = val.get("thresholds", last_thr)
        except Exception as e:
            print(f"  [tune] failed ({type(e).__name__}: {e}); fixed-{cfg.THRESHOLD} fallback")
            val = validate(model, spec_extractor, val_loader, val_criterion, device,
                           val_annotations, file_start_dts, threshold=cfg.THRESHOLD)
        eval_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        ema.restore(model)

        macro = sum(val["per_class"][c]["f1"] for c in cfg.CALL_TYPES_3) / 3
        improved = macro > best_macro
        if improved:
            best_macro = macro

        print(f"[NAVE] epoch {epoch:2d}/{epochs} ({time.time()-t0:.0f}s)"
              f"{' *** new best' if improved else ''}")
        print(f"  train {train_loss:.4f}  val {val['loss']:.4f}  "
              f"MACRO {macro:.3f}  THR {[round(t,3) for t in val.get('thresholds', [])]}")
        for name in cfg.CALL_TYPES_3:
            pc = val["per_class"][name]
            print(f"    {name.upper():6} P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f}")

        ckpt = {"model_state_dict": eval_state, "epoch": epoch, "seed": seed,
                "macro_f1": macro, "thresholds": val.get("thresholds"),
                "model": "NAVE", "ema_decay": cfg.EMA_DECAY}
        torch.save(ckpt, run_dir / f"nave_epoch_{epoch:02d}.pt")
        if improved:
            torch.save(ckpt, run_dir / "nave_best.pt")

    print(f"\n[NAVE] best tuned macro F1 = {best_macro:.3f}   ->  {run_dir/'nave_best.pt'}")


if __name__ == "__main__":
    main()
