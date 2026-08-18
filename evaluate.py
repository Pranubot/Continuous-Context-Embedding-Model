"""Evaluate every available checkpoint on the test set and print the results table.

Covers the three required conditions (direct, context, shuffled-embedding
check) plus any trained ablations (capacity-matched direct, random-init LLM).

The shuffle check uses derangements (no example keeps its own embedding)
across three seeds, without retraining, per the spec.
"""

from pathlib import Path

import numpy as np
import torch

import common
from context_model import ContextClassifier
from models import CapacityMatchedClassifier, DirectClassifier

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
SHUFFLE_SEEDS = [0, 1, 2]
SEED = 42


def derangement(n: int, seed: int) -> np.ndarray:
    """Random permutation with no fixed points."""
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def load_direct(name: str) -> torch.nn.Module | None:
    path = CHECKPOINT_DIR / f"{name}.pt"
    if not path.exists():
        return None
    model = (CapacityMatchedClassifier() if name == "direct_matched" else DirectClassifier())
    model.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
    return model.to(common.DEVICE)


def load_context(name: str) -> ContextClassifier | None:
    path = CHECKPOINT_DIR / f"{name}.pt"
    if not path.exists():
        return None
    checkpoint = torch.load(path, weights_only=True)
    # Random-LLM weights are not checkpointed; reseeding before construction
    # reproduces the same random initialization used in training.
    common.set_seed(checkpoint["seed"])
    model = ContextClassifier(random_llm=checkpoint["random_llm"])
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if unexpected or any(not k.startswith("llm.") for k in missing):
        raise ValueError(f"Bad checkpoint {name}: missing={missing} unexpected={unexpected}")
    return model.to(common.DEVICE)


@torch.no_grad()
def context_predictions(model: ContextClassifier, X: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
    """Test predictions plus the per-example projected embeddings."""
    model.eval()
    preds, embeddings = [], []
    for start in range(0, len(X), 128):
        vec = model.project(X[start : start + 128].to(common.DEVICE))
        preds.append(model.classify_embedding(vec).argmax(dim=1).cpu())
        embeddings.append(vec.cpu())
    return torch.cat(preds).numpy(), torch.cat(embeddings)


@torch.no_grad()
def shuffled_f1(model: ContextClassifier, embeddings: torch.Tensor, y: np.ndarray, seed: int) -> float:
    shuffled = embeddings[derangement(len(embeddings), seed)]
    preds = []
    for start in range(0, len(shuffled), 128):
        vec = shuffled[start : start + 128].to(common.DEVICE)
        preds.append(model.classify_embedding(vec).argmax(dim=1).cpu())
    return common.macro_f1(y, torch.cat(preds).numpy())


def main() -> None:
    splits = common.get_tensor_splits()
    test_X, test_y = splits["test"]
    y_true = test_y.numpy()

    rows: list[tuple[str, str]] = []

    majority = np.bincount(splits["fit"][1].numpy()).argmax()
    majority_f1 = common.macro_f1(y_true, np.full_like(y_true, majority))
    rows.append(("Constant-majority floor", f"{majority_f1:.4f}"))

    for name, label in [("direct", "Direct sensor classifier"),
                        ("direct_matched", "Capacity-matched direct (ablation)")]:
        model = load_direct(name)
        if model is not None:
            f1 = common.macro_f1(y_true, common.predict(model, test_X, test_y))
            rows.append((label, f"{f1:.4f}"))

    context = load_context("context")
    if context is not None:
        preds, embeddings = context_predictions(context, test_X)
        rows.append(("Context-embedding model", f"{common.macro_f1(y_true, preds):.4f}"))

        scores = [shuffled_f1(context, embeddings, y_true, s) for s in SHUFFLE_SEEDS]
        rows.append((f"Context, shuffled embeddings ({len(SHUFFLE_SEEDS)} derangements)",
                     f"{np.mean(scores):.4f} ± {np.std(scores):.4f}"))

    random_llm = load_context("context_random_llm")
    if random_llm is not None:
        preds, _ = context_predictions(random_llm, test_X)
        rows.append(("Random-init frozen LLM (ablation)", f"{common.macro_f1(y_true, preds):.4f}"))

    width = max(len(label) for label, _ in rows)
    print(f"\n{'Condition':<{width}}  Macro-F1")
    print("-" * (width + 20))
    for label, value in rows:
        print(f"{label:<{width}}  {value}")


if __name__ == "__main__":
    main()
