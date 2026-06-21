"""
NAVE -- Normalized, Adaptive Conformer for Whale Vocalization-Event Detection
=============================================================================
Configuration: the single source of truth for every NAVE constant. Every other
``nave_*`` module imports this as ``cfg``.

Name mapping: N = Normalized (PCEN channel), A = Adaptive (frequency-dynamic
FDY convolutions in the stem), Conformer backbone, for whale Vocalization-Event
detection.
"""

from pathlib import Path

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
DATA_ROOT = Path("/home/matthias-nagl/BioDCASE/task/2026_BioDCASE_development_set/")
OUTPUT_DIR = Path("./runs")

TRAIN_DATASETS = [
    "ballenyislands2015", "casey2014", "elephantisland2013", "elephantisland2014",
    "greenwich2015", "kerguelen2005", "maudrise2014", "rosssea2014",
]
VAL_DATASETS = ["casey2017", "kerguelen2014", "kerguelen2015"]
# Quarantined blind test sites (never imported into train/val; inference only).
TEST_DATASETS = ["kerguelen2020", "ddu2021"]

# ----------------------------------------------------------------------
# Front end (phase-aware STFT + PCEN channel)
# ----------------------------------------------------------------------
SAMPLE_RATE = 250
N_FFT = 256
WIN_LENGTH = 256
HOP_LENGTH = 5
FRAME_STRIDE_S = 0.02            # 20 ms / frame (HOP_LENGTH / SAMPLE_RATE)
NORM_FEATURES = "demean"        # per-frequency complex mean subtraction (ch 0-2)
DUAL_RESOLUTION = False         # NAVE uses the single-resolution 3-ch base + PCEN

# Fixed PCEN (4th channel) -- librosa-style defaults, parameter-free.
PCEN_ALPHA = 0.98
PCEN_DELTA = 2.0
PCEN_POWER = 0.5
PCEN_SMOOTH = 0.025
PCEN_EPS = 1e-6
PCEN_EMA_TAPS = 512

FEAT_CHANNELS = 4               # [demeaned |S|, cos phi, sin phi, PCEN]

# ----------------------------------------------------------------------
# Classes (7 fine -> 3 coarse). NAVE trains directly on the 3 coarse classes.
# ----------------------------------------------------------------------
CALL_TYPES_7 = ["bma", "bmb", "bmz", "bmd", "bpd", "bp20", "bp20plus"]
CALL_TYPES_3 = ["bmabz", "d", "bp"]
N_CLASSES = 3
COLLAPSE_MAP = {
    "bma": "bmabz", "bmb": "bmabz", "bmz": "bmabz",
    "bmd": "d", "bpd": "d",
    "bp20": "bp", "bp20plus": "bp",
}
# Mutable global: the post-processing path flips this True while it labels
# detections in the coarse space, then restores it. Shared module object, so
# the toggle is observed everywhere consistently.
USE_3CLASS = True

# ----------------------------------------------------------------------
# Segmentation / collar
# ----------------------------------------------------------------------
TRAIN_SEGMENT_S = 30.0
EVAL_SEGMENT_S = 30.0
EVAL_OVERLAP_S = 2.0
COLLAR_MIN_S = 1.0
COLLAR_MAX_S = 5.0
MIN_CALL_DURATION_S = 0.5
MAX_CALL_DURATION_S = 30.0
NEG_RATIO = 1.0                 # negatives per positive segment, resampled / epoch

# ----------------------------------------------------------------------
# Stem (WhaleVAD CNN feature extractor, inherited)
# ----------------------------------------------------------------------
FILTERBANK_OUT_CH = 64
FEAT_EXTRACTOR_CH = 128
BOTTLENECK_CH = 64
BOTTLENECK_DROPOUT = 0.1
AGG_DROPOUT = 0.2

# ----------------------------------------------------------------------
# NAVE architecture
# ----------------------------------------------------------------------
FDY_TARGETS = ("filterbank", "feat0")   # frequency-dynamic convs in the stem
FDY_BASIS = 4                           # K basis kernels
FDY_TEMP = 1.0                          # attention softmax temperature

D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 4
FFN_MULT = 4
CONV_KERNEL = 129                       # wide depthwise kernel
DROPOUT = 0.1

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
OPTIMIZER = "radam"             # plain RAdam, constant LR, no warmup
EPOCHS = 40
BATCH_SIZE = 32
LR = 5e-5
WEIGHT_DECAY = 1e-3
BETA1 = 0.9
BETA2 = 0.999
GRAD_CLIP = 1.0
EMA_DECAY = 0.999               # EMA-of-weights teacher (validated + checkpointed)
RESAMPLE_EVERY = 1              # resample the negative pool every N epochs
SEED = 42
NUM_WORKERS = 16
SELECT_BY = "macro"             # checkpoint selection metric (tuned macro F1)
POS_WEIGHT = None               # masked weighted BCE; None = unweighted positives

# ----------------------------------------------------------------------
# Post-processing (structural params fixed; only per-class thresholds tuned)
# ----------------------------------------------------------------------
THRESHOLD = 0.3                 # fixed-threshold baseline; tuned per class at eval
SMOOTH_KERNEL_MS = 500          # temporal median filter on probabilities
MERGE_GAP_S = 0.5               # merge same-class events closer than this
POST_MIN_DUR_S = 0.5
POST_MAX_DUR_S = 30.0
IOU_THRESHOLD = 0.3             # event-matching IoU for metrics

def n_classes() -> int:
    return N_CLASSES


def n_feat_channels() -> int:
    return FEAT_CHANNELS


def class_names() -> list[str]:
    """Ordered class labels currently in use. NAVE uses the 3 coarse classes
    (``USE_3CLASS`` is always True); the 7-class branch is kept only so the
    shared post-processing helpers stay general."""
    return list(CALL_TYPES_3) if USE_3CLASS else list(CALL_TYPES_7)


def class_to_idx() -> dict[str, int]:
    """Mapping from class name to zero-based output index."""
    return {c: i for i, c in enumerate(class_names())}
