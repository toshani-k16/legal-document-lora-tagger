
"""Tests for the Legal Document LoRA Tagger."""

import pytest
import app.core as core


# Milestone 1: Prepare a dataset of legal clauses for tagging

def test_milestone_1():
    result = core.milestone_1()

    assert isinstance(result, dict)
    assert "train" in result
    assert "validation" in result
    assert "test" in result
    assert "num_labels" in result

    assert len(result["train"]) > 0
    assert len(result["validation"]) > 0
    assert len(result["test"]) > 0
    assert result["num_labels"] > 0


# Milestone 2: Extend GPT-2 tokenizer and model for custom tags

def test_milestone_2(monkeypatch):

    class FakeTokenizer:
        def __init__(self):
            self.added_tokens = []

        def add_special_tokens(self, data):
            self.added_tokens = data["additional_special_tokens"]
            return len(self.added_tokens)

        def __len__(self):
            return 50257 + len(self.added_tokens)

    class FakeModel:
        def __init__(self):
            self.resized_to = None

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


# Milestone 3: Implement LoRA adapter and inject into GPT-2

@pytest.mark.skip(reason="not started yet")
def test_milestone_3():
    raise NotImplementedError("TODO(milestone 3)")


# Milestone 4: Fine-tune GPT-2 with LoRA

@pytest.mark.skip(reason="not started yet")
def test_milestone_4():
    raise NotImplementedError("TODO(milestone 4)")


# Milestone 5: Merge LoRA adapter and perform inference

@pytest.mark.skip(reason="not started yet")
def test_milestone_5():
    raise NotImplementedError("TODO(milestone 5)")


# Milestone 6: Evaluate tagging accuracy and model efficiency

@pytest.mark.skip(reason="not started yet")
def test_milestone_6():
    raise NotImplementedError("TODO(milestone 6)")
