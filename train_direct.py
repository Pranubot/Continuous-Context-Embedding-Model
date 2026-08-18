"""Train the direct sensor classifier (required baseline).

Usage:
    python train_direct.py             # encoder -> linear head
    python train_direct.py --matched   # capacity-matched ablation
"""

import argparse
from pathlib import Path

import torch

import common
from models import CapacityMatchedClassifier, DirectClassifier, count_trainable

SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched", action="store_true",
                        help="train the capacity-matched ablation instead")
    args = parser.parse_args()

    common.set_seed(SEED)
    splits = common.get_tensor_splits()

    model = (CapacityMatchedClassifier() if args.matched else DirectClassifier()).to(common.DEVICE)
    name = "direct_matched" if args.matched else "direct"
    print(f"model: {name}  trainable params: {count_trainable(model):,}")

    result = common.train(model, splits, lr=1e-3, epochs=30, batch_size=128, seed=SEED)

    test_pred = common.predict(model, *splits["test"])
    test_true = splits["test"][1].numpy()
    test_f1 = common.macro_f1(test_true, test_pred)

    print(f"\nbest val macro-F1: {result['val_f1']:.4f} (epoch {result['epoch'] + 1})")
    print(f"test macro-F1:     {test_f1:.4f}")
    print("\ntest confusion matrix (rows = true):")
    print(common.format_confusion(common.confusion_matrix(test_true, test_pred)))

    out_dir = Path(__file__).parent / "checkpoints"
    out_dir.mkdir(exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "test_f1": test_f1,
         "val_f1": result["val_f1"], "seed": SEED},
        out_dir / f"{name}.pt",
    )
    print(f"\nsaved checkpoints/{name}.pt")


if __name__ == "__main__":
    main()
