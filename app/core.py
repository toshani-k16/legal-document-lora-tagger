
"""Core implementation for the Legal Document LoRA Tagger."""

from datasets import load_dataset


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


# Remaining milestones will be implemented next.
def milestone_2():
    raise NotImplementedError("TODO(milestone 2)")


def milestone_3():
    raise NotImplementedError("TODO(milestone 3)")


def milestone_4():
    raise NotImplementedError("TODO(milestone 4)")


def milestone_5():
    raise NotImplementedError("TODO(milestone 5)")


def milestone_6():
    raise NotImplementedError("TODO(milestone 6)")
