
"""Tests for the Legal Document LoRA Tagger."""

import pytest
import torch
import torch.nn as nn
import app.core as core


# MILESTONE 1: Dataset preparation
def test_milestone_1():
    assert callable(core.milestone_1)


# MILESTONE 2: Tokenizer extension
def test_milestone_2():
    assert callable(core.milestone_2)


# MILESTONE 3: LoRA implementation
def test_milestone_3():
    base = nn.Linear(10, 5)

    adapter = core.LoRALayer(
        base,
        rank=2,
        alpha=4,
        dropout=0.0
    )

    x = torch.randn(3, 10)
    output = adapter(x)

    assert output.shape == (3, 5)

    assert all(
        not p.requires_grad
        for p in adapter.base.parameters()
    )

    assert adapter.lora_A.requires_grad
    assert adapter.lora_B.requires_grad


# MILESTONE 4: Fine-tuning
def test_milestone_4():
    assert callable(core.milestone_4)


# MILESTONE 5: Merge and inference
def test_milestone_5():
    assert callable(core.milestone_5)


# MILESTONE 6: Evaluation
def test_milestone_6():
    assert callable(core.milestone_6)

    result = core.milestone_6

    assert result is not None
    assert callable(result)
