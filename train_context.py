"""Train the context-embedding model (frozen SmolLM2).

Usage:
    python train_context.py               # pretrained frozen LLM
    python train_context.py --random-llm  # ablation: random-init frozen LLM
"""

import argparse
from pathlib import Path

import torch

import common
from context_model import ContextClassifier
from models import count_trainable

SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-llm", action="store_true",
                        help="use a randomly initialized frozen LLM instead of pretrained")
    args = parser.parse_args()

    common.set_seed(SEED)
    splits = common.get_tensor_splits()

    model = ContextClassifier(random_llm=args.random_llm).to(common.DEVICE)
    name = "context_random_llm" if args.random_llm else "context"
    print(f"model: {name}  trainable params: {count_trainable(model):,}")

    result = common.train(
        model, splits, lr=3e-4, epochs=25, batch_size=64, seed=SEED, warmup_epochs=2
    )

    test_pred = common.predict(model, *splits["test"], batch_size=128)
    test_true = splits["test"][1].numpy()
    test_f1 = common.macro_f1(test_true, test_pred)

    print(f"\nbest val macro-F1: {result['val_f1']:.4f} (epoch {result['epoch'] + 1})")
    print(f"test macro-F1:     {test_f1:.4f}")
    print("\ntest confusion matrix (rows = true):")
    print(common.format_confusion(common.confusion_matrix(test_true, test_pred)))

    out_dir = Path(__file__).parent / "checkpoints"
    out_dir.mkdir(exist_ok=True)
    torch.save(
        {"state_dict": model.trainable_state_dict(), "test_f1": test_f1,
         "val_f1": result["val_f1"], "seed": SEED, "random_llm": args.random_llm},
        out_dir / f"{name}.pt",
    )
    print(f"\nsaved checkpoints/{name}.pt")


if __name__ == "__main__":
    main()
