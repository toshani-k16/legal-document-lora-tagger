
"""Tests for the Legal Document LoRA Tagger."""

import pytest

from app.core import milestone_1


# Milestone 1: Prepare a dataset of legal clauses for tagging
def test_milestone_1():
    result = milestone_1()

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
@pytest.mark.skip(reason="not started yet")
def test_milestone_2():
    raise NotImplementedError("TODO(milestone 2)")


# Milestone 3: Implement LoRA adapter and inject into GPT-2
@pytest.mark.skip(reason="not started yet")
def test_milestone_3():
    raise NotImplementedError("TODO(milestone 3)")


# Milestone 4: Fine-tune GPT-2 with LoRA on the clause dataset
@pytest.mark.skip(reason="not started yet")
def test_milestone_4():
    raise NotImplementedError("TODO(milestone 4)")


# Milestone 5: Merge LoRA adapter and perform inference on new clauses
@pytest.mark.skip(reason="not started yet")
def test_milestone_5():
    raise NotImplementedError("TODO(milestone 5)")


# Milestone 6: Evaluate tagging accuracy and model efficiency
@pytest.mark.skip(reason="not started yet")
def test_milestone_6():
    raise NotImplementedError("TODO(milestone 6)")
