"""Data pipeline for the UCI HAR inertial signals.

Loads the nine raw windowed signals (128 x 9 per example), applies a
subject-wise train/val split, and normalizes with global per-channel
statistics computed on the training subjects only.
"""

from pathlib import Path

import numpy as np

DATASET_DIR = Path(__file__).parent / "Dataset"
CACHE_DIR = Path(__file__).parent / "cache"

SEED = 42
NUM_VAL_SUBJECTS = 4

SIGNALS = [
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]

LABEL_NAMES = [
    "walking", "walking_upstairs", "walking_downstairs",
    "sitting", "standing", "laying",
]


def _load_split_raw(split: str) -> dict[str, np.ndarray]:
    """Load one official split ("train" or "test") as float32 arrays.

    Returns {"X": (N, 9, 128), "y": (N,) in [0, 5], "subject": (N,)}.
    Results are cached to .npz because np.loadtxt on the text files is slow.
    """
    cache_file = CACHE_DIR / f"{split}.npz"
    if cache_file.exists():
        cached = np.load(cache_file)
        return {key: cached[key] for key in ("X", "y", "subject")}

    split_dir = DATASET_DIR / split
    signals = [
        np.loadtxt(split_dir / "Inertial Signals" / f"{name}_{split}.txt", dtype=np.float32)
        for name in SIGNALS
    ]
    X = np.stack(signals, axis=1)  # (N, 9, 128)
    y = np.loadtxt(split_dir / f"y_{split}.txt", dtype=np.int64) - 1
    subject = np.loadtxt(split_dir / f"subject_{split}.txt", dtype=np.int64)

    if X.shape[1:] != (9, 128):
        raise ValueError(f"Unexpected signal shape {X.shape} for split '{split}'")
    if not (len(X) == len(y) == len(subject)):
        raise ValueError(f"Row-count mismatch in split '{split}'")
    if y.min() < 0 or y.max() > 5:
        raise ValueError(f"Labels out of range in split '{split}'")

    CACHE_DIR.mkdir(exist_ok=True)
    np.savez(cache_file, X=X, y=y, subject=subject)
    return {"X": X, "y": y, "subject": subject}


def val_subjects() -> list[int]:
    """Deterministically pick validation subjects from the training split."""
    train_subjects = sorted(set(_load_split_raw("train")["subject"].tolist()))
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(train_subjects, size=NUM_VAL_SUBJECTS, replace=False)
    return sorted(int(s) for s in chosen)


def get_splits() -> dict[str, dict[str, np.ndarray]]:
    """Return normalized fit/val/test splits.

    "fit" is the training data actually used for gradient updates (train subjects minus validation subjects). 
    Normalization statistics are per-channel mean/std computed on "fit" only and applied everywhere.
    """
    train = _load_split_raw("train")
    test = _load_split_raw("test")

    held_out = val_subjects()
    val_mask = np.isin(train["subject"], held_out)

    splits = {
        "fit": {key: arr[~val_mask] for key, arr in train.items()},
        "val": {key: arr[val_mask] for key, arr in train.items()},
        "test": test,
    }

    mean = splits["fit"]["X"].mean(axis=(0, 2), keepdims=True)  # (1, 9, 1)
    std = splits["fit"]["X"].std(axis=(0, 2), keepdims=True)

    return {
        name: {**split, "X": (split["X"] - mean) / std}
        for name, split in splits.items()
    }


def _describe(name: str, split: dict[str, np.ndarray]) -> None:
    counts = np.bincount(split["y"], minlength=6)
    subjects = sorted(set(split["subject"].tolist()))
    print(f"{name:5s} X={split['X'].shape}  subjects={subjects}")
    print(f"      labels: " + ", ".join(f"{n}={c}" for n, c in zip(LABEL_NAMES, counts)))


if __name__ == "__main__":
    raw_train = _load_split_raw("train")

    # Gravity sanity check on raw (unnormalized) data: total_acc - body_acc | should be a slowly varying ~1 g vector.
    gravity = raw_train["X"][:, 0:3] - raw_train["X"][:, 3:6]  # (N, 3, 128)
    gravity_norm = np.linalg.norm(gravity, axis=1)  # (N, 128)
    print(f"gravity |g|: mean={gravity_norm.mean():.4f} g, std={gravity_norm.std():.4f} g")
    print(f"gravity temporal variation (mean within-window std): "
          f"{gravity.std(axis=2).mean():.4f} g")

    print(f"\nval subjects: {val_subjects()}")
    splits = get_splits()
    for name, split in splits.items():
        _describe(name, split)

    fit_x = splits["fit"]["X"]
    print(f"\nnormalized fit: per-channel mean max |.| = "
          f"{np.abs(fit_x.mean(axis=(0, 2))).max():.2e}, "
          f"std range = [{fit_x.std(axis=(0, 2)).min():.3f}, "
          f"{fit_x.std(axis=(0, 2)).max():.3f}]")
