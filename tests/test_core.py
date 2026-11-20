
"""Tests for the Legal Document LoRA Tagger."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import app.core as core


# MILESTONE 1: Prepare legal clause dataset
def test_milestone_1():
    result = core.milestone_1()

    assert isinstance(result, dict)
    assert "train" in result
    assert "validation" in result
    assert "test" in result
    assert result["num_labels"] > 0


# MILESTONE 2: Extend GPT-2 tokenizer
def test_milestone_2(monkeypatch):

    class FakeTokenizer:
        def __init__(self):
            self.added_tokens = []

        def add_special_tokens(self, data):
            self.added_tokens = data["additional_special_tokens"]

        def __len__(self):
            return 50257 + len(self.added_tokens)

    class FakeModel:
        def resize_token_embeddings(self, size):
            self.resized_to = size

    tokenizer = FakeTokenizer()
    model = FakeModel()

    monkeypatch.setattr(
        core.AutoTokenizer,
        "from_pretrained",
        lambda name: tokenizer
    )

    monkeypatch.setattr(
        core.AutoModelForCausalLM,
        "from_pretrained",
        lambda name: model
    )

    result = core.milestone_2()

    assert result["tokenizer"] is tokenizer
    assert result["model"] is model
    assert len(tokenizer.added_tokens) > 0
    assert model.resized_to == len(tokenizer)


# MILESTONE 3: Implement LoRA adapter
def test_milestone_3():

    base_layer = nn.Linear(10, 5)

    adapter = core.LoRALayer(
        base_layer,
        rank=2,
        alpha=4,
        dropout=0.0
    )

    x = torch.randn(3, 10)
    output = adapter(x)

    assert output.shape == (3, 5)

    assert all(
        not parameter.requires_grad
        for parameter in adapter.base.parameters()
    )

    assert adapter.lora_A.requires_grad
    assert adapter.lora_B.requires_grad


# MILESTONE 4: Fine-tune GPT-2 with LoRA
def test_milestone_4():

    adapter_path = Path("outputs/legal_lora/adapter.pt")

    assert adapter_path.exists(), (
        "LoRA adapter file was not found. Run milestone_4 first."
    )

    assert adapter_path.stat().st_size > 0


# MILESTONE 5: Merge LoRA and perform inference
def test_milestone_5():

    clause = (
        "This agreement shall be governed by the laws "
        "of the State of California."
    )

    result = core.milestone_5(clause)

    assert result is not None


# MILESTONE 6: Evaluate tagging accuracy
@pytest.mark.skip(reason="Start after milestone 5")
def test_milestone_6():
    raise NotImplementedError("TODO(milestone 6)")
