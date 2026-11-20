
"""Core implementation for the Legal Document LoRA Tagger."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# --- Milestone 1: Prepare legal clause dataset

def milestone_1():
    """Load the LEDGAR legal clause dataset."""

    dataset = load_dataset("coastalcph/lex_glue", "ledgar")

    return {
        "train": dataset["train"],
        "validation": dataset["validation"],
        "test": dataset["test"],
        "num_labels": len(dataset["train"].features["label"].names),
    }


# --- Milestone 2: Extend GPT-2 tokenizer and model

def milestone_2():
    """Add custom legal tags to GPT-2."""

    model_name = "gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    legal_tags = [
        "<Governing_Law>",
        "<Indemnity>",
        "<Confidentiality>",
        "<Termination>",
        "<Liability>",
    ]

    tokenizer.add_special_tokens({
        "additional_special_tokens": legal_tags
    })

    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.resize_token_embeddings(len(tokenizer))

    return {
        "tokenizer": tokenizer,
        "model": model,
        "legal_tags": legal_tags,
    }


# --- Milestone 3: Implement LoRA adapter

class LoRALayer(nn.Module):
    """Low-Rank Adaptation wrapper for a linear-like layer."""

    def __init__(self, base, rank=2, alpha=4, dropout=0.0):
        super().__init__()

        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Support nn.Linear and Hugging Face GPT-2 Conv1D layers.
        if hasattr(base, "in_features"):
            input_size = base.in_features
            output_size = base.out_features
        else:
            input_size = base.weight.shape[0]
            output_size = base.weight.shape[1]

        # Freeze original pretrained weights.
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        # Trainable low-rank matrices.
        self.lora_A = nn.Parameter(
            torch.empty(rank, input_size)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(output_size, rank)
        )

        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Combine frozen base output with trainable LoRA update."""

        base_output = self.base(x)

        lora_output = F.linear(
            self.dropout(x),
            self.lora_A
        )

        lora_output = F.linear(
            lora_output,
            self.lora_B
        )

        return base_output + self.scaling * lora_output


def milestone_3(model=None):
    """Inject LoRA adapters into GPT-2 attention projections."""

    if model is None:
        model = AutoModelForCausalLM.from_pretrained("gpt2")

    # GPT-2 uses a fused attention projection named c_attn.
    for block in model.transformer.h:
        block.attn.c_attn = LoRALayer(
            block.attn.c_attn,
            rank=8,
            alpha=16,
            dropout=0.1
        )

    return model


# --- Remaining milestones

def milestone_4():
    raise NotImplementedError("TODO(milestone 4)")


def milestone_5():
    raise NotImplementedError("TODO(milestone 5)")


def milestone_6():
    raise NotImplementedError("TODO(milestone 6)")
