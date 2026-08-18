"""Context-embedding model: sensor encoder -> projector -> frozen SmolLM2 -> head.

The projected sensor vector replaces <SENSOR> in the spec prompt. It is
inserted directly into the input-embedding sequence (inputs_embeds); sensor
values are never converted to text.
"""

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from models import LLM_DIM, NUM_CLASSES, Projector, SensorEncoder

LLM_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

PROMPT_PREFIX = (
    "Classify the activity as walking, walking upstairs, walking downstairs, "
    "sitting, standing, or laying.\n\nSensor context: "
)
PROMPT_SUFFIX = "\nActivity:"


class ContextClassifier(nn.Module):
    """Trainable encoder + projector + linear head around a frozen LLM.

    Gradients flow through the frozen transformer (no torch.no_grad), but only
    encoder, projector, and head receive updates.
    """

    def __init__(self, random_llm: bool = False) -> None:
        super().__init__()
        if random_llm:
            config = AutoConfig.from_pretrained(LLM_NAME)
            self.llm = AutoModel.from_config(config).float() 
        else:
            self.llm = AutoModel.from_pretrained(LLM_NAME, dtype=torch.float32)
        self.llm.requires_grad_(False)
        self.llm.eval()

        embeddings = self.llm.get_input_embeddings()
        target_norm = embeddings.weight.norm(dim=1).mean().item()

        self.encoder = SensorEncoder()
        self.projector = Projector(target_norm=target_norm)
        self.head = nn.Linear(LLM_DIM, NUM_CLASSES)

        # The prompt is identical for every example, so its embeddings are constant; precompute them once. 
        # Raw text per the spec: no chat template and no special tokens (SmolLM2's BOS is the chat delimiter <|im_start|>, which does not belong in a raw prompt).
        tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
        prefix_ids = tokenizer(PROMPT_PREFIX, add_special_tokens=False).input_ids
        suffix_ids = tokenizer(PROMPT_SUFFIX, add_special_tokens=False).input_ids

        reconstructed = tokenizer.decode(prefix_ids) + "<SENSOR>" + tokenizer.decode(suffix_ids)
        expected = PROMPT_PREFIX + "<SENSOR>" + PROMPT_SUFFIX
        if reconstructed != expected:
            raise ValueError(f"Prompt round-trip mismatch:\n{reconstructed!r}\n{expected!r}")

        with torch.no_grad():
            to_embed = lambda ids: embeddings(torch.tensor([ids])).detach()
            self.register_buffer("prefix_embeds", to_embed(prefix_ids), persistent=False)
            self.register_buffer("suffix_embeds", to_embed(suffix_ids), persistent=False)

    def train(self, mode: bool = True) -> "ContextClassifier":
        """Keep the frozen LLM in eval mode regardless of outer train/eval."""
        super().train(mode)
        self.llm.eval()
        return self

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Sensor window -> projected context embedding, (batch, 960)."""
        return self.projector(self.encoder(x))

    def classify_embedding(self, sensor_vec: torch.Tensor) -> torch.Tensor:
        """Projected embedding -> logits. Split out for the shuffle check."""
        batch = sensor_vec.shape[0]
        inputs_embeds = torch.cat(
            [
                self.prefix_embeds.expand(batch, -1, -1),
                sensor_vec.unsqueeze(1),
                self.suffix_embeds.expand(batch, -1, -1),
            ],
            dim=1,
        )
        hidden = self.llm(inputs_embeds=inputs_embeds).last_hidden_state
        return self.head(hidden[:, -1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_embedding(self.project(x))

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Checkpoint only what we train; the frozen LLM is reloadable by name."""
        return {
            k: v for k, v in self.state_dict().items()
            if k.startswith(("encoder.", "projector.", "head."))
        }


if __name__ == "__main__":
    model = ContextClassifier()
    print("prompt round-trip check passed:")
    print(repr(PROMPT_PREFIX + "<SENSOR>" + PROMPT_SUFFIX))
    print(f"prefix tokens: {model.prefix_embeds.shape[1]}, "
          f"suffix tokens: {model.suffix_embeds.shape[1]}, sensor tokens: 1")
    with torch.no_grad():
        logits = model(torch.randn(2, 9, 128))
    print(f"logits shape: {tuple(logits.shape)}")
