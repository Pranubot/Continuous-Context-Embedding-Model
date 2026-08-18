"""Shared training/evaluation utilities for both the direct and context models."""

import random

import numpy as np
import torch
from torch import nn

import data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_tensor_splits() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Splits from data.get_splits() as CPU tensors: {name: (X, y)}."""
    return {
        name: (torch.from_numpy(split["X"]), torch.from_numpy(split["y"]))
        for name, split in data.get_splits().items()
    }


def iter_batches(
    X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool, seed: int = 0
):
    order = (
        torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
        if shuffle
        else torch.arange(len(X))
    )
    for start in range(0, len(X), batch_size):
        idx = order[start : start + batch_size]
        yield X[idx].to(DEVICE), y[idx].to(DEVICE)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 6) -> float:
    scores = []
    for c in range(num_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(scores))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 6) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def format_confusion(matrix: np.ndarray) -> str:
    width = 8
    header = " " * 20 + "".join(f"{name[:width]:>{width + 1}}" for name in data.LABEL_NAMES)
    rows = [
        f"{name:>19} " + "".join(f"{v:>{width + 1}}" for v in row)
        for name, row in zip(data.LABEL_NAMES, matrix)
    ]
    return "\n".join([header, *rows])


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    model.eval()
    preds = [
        model(xb).argmax(dim=1).cpu()
        for xb, _ in iter_batches(X, y, batch_size, shuffle=False)
    ]
    return torch.cat(preds).numpy()


def train(
    model: nn.Module,
    splits: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 0,
) -> dict:
    """Train with AdamW + cosine schedule, keep the best-val-macro-F1 state

    Returns {"model": model (best state loaded), "val_f1": float, "epoch": int}.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    fit_X, fit_y = splits["fit"]
    steps_per_epoch = (len(fit_X) + batch_size - 1) // batch_size
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.CrossEntropyLoss()

    best = {"val_f1": -1.0, "state": None, "epoch": -1}
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in iter_batches(fit_X, fit_y, batch_size, shuffle=True, seed=seed + epoch):
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(xb)

        val_pred = predict(model, *splits["val"])
        val_f1 = macro_f1(splits["val"][1].numpy(), val_pred)
        marker = ""
        if val_f1 > best["val_f1"]:
            trained_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            best = {"val_f1": val_f1, "state": trained_state, "epoch": epoch}
            marker = " *"
        print(f"epoch {epoch + 1:3d}/{epochs}  loss={total_loss / len(fit_X):.4f}  "
              f"val_f1={val_f1:.4f}{marker}")

    model.load_state_dict(best["state"])
    return {"model": model, "val_f1": best["val_f1"], "epoch": best["epoch"]}
