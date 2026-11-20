
"""Core implementation for the Legal Document LoRA Tagger."""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# --- Milestone 1: Prepare a dataset of legal clauses for tagging

def milestone_1():
    """Load and prepare the LEDGAR legal clause dataset."""

    dataset = load_dataset("coastalcph/lex_glue", "ledgar")

    train_data = dataset["train"]
    validation_data = dataset["validation"]
    test_data = dataset["test"]

    return {
        "train": train_data,
        "validation": validation_data,
        "test": test_data,
        "num_labels": len(dataset["train"].features["label"].names),
    }


# --- Milestone 2: Extend GPT-2 tokenizer and model for custom tags

def milestone_2():
    """Add custom legal tags to GPT-2 tokenizer and resize embeddings."""

    model_name = "gpt2"

    # Load pretrained GPT-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define custom legal clause tags
    legal_tags = [
        "<Governing_Law>",
        "<Indemnity>",
        "<Confidentiality>",
        "<Termination>",
        "<Liability>",
    ]

    # Add legal tags as special tokens
    tokenizer.add_special_tokens({
        "additional_special_tokens": legal_tags
    })

    # Load pretrained GPT-2 model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Resize embeddings to match the expanded tokenizer vocabulary
    model.resize_token_embeddings(len(tokenizer))

    return {
        "tokenizer": tokenizer,
        "model": model,
        "legal_tags": legal_tags,
    }


# --- Remaining milestones

def milestone_3():
    raise NotImplementedError("TODO(milestone 3)")


def milestone_4():
    raise NotImplementedError("TODO(milestone 4)")


def milestone_5():
    raise NotImplementedError("TODO(milestone 5)")


def milestone_6():
    raise NotImplementedError("TODO(milestone 6)")
