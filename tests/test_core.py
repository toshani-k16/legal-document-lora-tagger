
"""Tests for the Legal Document LoRA Tagger."""

import pytest
import torch
import torch.nn as nn
import app.core as core


# Milestone 1: Dataset preparation
def test_milestone_1():
    result = core.milestone_1()

    assert isinstance(result, dict)
    assert "train" in result
    assert "validation" in result
    assert "test" in result
    assert result["num_labels"] > 0


# Milestone 2: GPT-2 tokenizer extension
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


# Milestone 3: LoRA adapter
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


# Milestone 4: Fine-tune GPT-2 with LoRA
def test_milestone_4(tmp_path):
    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            batch_size = len(texts)
            return {
                "input_ids": torch.ones(batch_size, 4, dtype=torch.long),
                "attention_mask": torch.ones(batch_size, 4, dtype=torch.long),
            }

    class FakeOutput:
        def __init__(self, loss):
            self.loss = loss

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_parameter = nn.Parameter(torch.tensor(0.5))

        def forward(self, input_ids, attention_mask, labels):
            loss = self.lora_parameter ** 2
            return FakeOutput(loss)

    dataset = {
        "train": [
            {"text": "The agreement is governed by law.", "label": 0},
            {"text": "The parties agree to indemnify.", "label": 1},
        ]
    }

    model = FakeModel()
    tokenizer = FakeTokenizer()

    result = core.milestone_4(
        dataset=dataset,
        tokenizer=tokenizer,
        model=model,
        output_dir=str(tmp_path),
        epochs=1,
        batch_size=2,
    )

    assert result["model"] is model
    assert result["tokenizer"] is tokenizer
    assert len(result["losses"]) > 0
    assert (tmp_path / "adapter.pt").exists()


# Milestone 5: Merge LoRA and inference
@pytest.mark.skip(reason="not started yet")
def test_milestone_5():
    raise NotImplementedError("TODO(milestone 5)")


# Milestone 6: Evaluation
@pytest.mark.skip(reason="not started yet")
def test_milestone_6():
    raise NotImplementedError("TODO(milestone 6)")
