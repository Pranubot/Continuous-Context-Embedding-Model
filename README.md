# Sensor Context Encoder Challenge

Connects a window of raw inertial-sensor data (UCI HAR, 128 × 9) to a frozen
language model (SmolLM2-360M-Instruct) through one continuous context
embedding, and compares that against a direct sensor classifier.

## Setup

Requires Python ≥ 3.10 with CUDA-enabled PyTorch. Dependencies:

```
pip install torch numpy transformers
```

(Developed with Python 3.12, torch 2.5.1+cu121, transformers 5.15.0, numpy. SmolLM2-360M-Instruct 
downloads automatically from the Hugging Face Hub on first run, ~1.4 GB.)

The UCI HAR dataset must be in `Dataset/` (as shipped with this repo):
`Dataset/{train,test}/Inertial Signals/*.txt` plus the label/subject files.

## Reproduce

All runs use **seed 42**. Validation subjects (3, 16, 22, 23) are drawn
deterministically from the training split; the official subject-wise
train/test split is untouched, and per-channel normalization statistics come
from the training subjects only.

```
python data.py                       # sanity checks: shapes, splits, gravity
python models.py                     # trainable parameter counts
python train_direct.py               # condition 1: direct classifier
python train_context.py              # condition 2: context-embedding model
python train_direct.py --matched     # ablation: capacity-matched direct
python train_context.py --random-llm # ablation: random-init frozen LLM
python evaluate.py                   # results table incl. shuffle check (condition 3)
```

`evaluate.py` re-evaluates every checkpoint found in `checkpoints/` on the
test set and runs the sensor-dependence check: projected test embeddings are
permuted with a derangement (no example keeps its own embedding) across three
seeds, without retraining.

## Results

Test-set macro-F1, one run per condition, seed 42:

| Condition | Macro-F1 |
|---|---|
| Direct sensor classifier | 0.9111 |
| Context-embedding model | 0.9037 |
| Context model with shuffled embeddings | 0.1661 ± 0.0032 |
| Capacity-matched direct (ablation) | 0.9112 |
| Random-init frozen LLM (ablation) | 0.9090 |
| Constant-majority floor | 0.0514 |

See `technical doc.pdf' for design, interpretation, limitations, and the
recommendation.

## Files

- `data.py` — loading, subject-wise splits, global normalization
- `models.py` — CNN sensor encoder, projector, heads, parameter counts
- `context_model.py` — frozen-LLM wrapper (prompt assembly via `inputs_embeds`)
- `common.py` — training loop, metrics
- `train_direct.py`, `train_context.py` — training entry points
- `evaluate.py` — results table and shuffled-embedding check
