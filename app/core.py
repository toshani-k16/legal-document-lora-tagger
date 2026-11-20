
"""Core implementation for the Legal Document LoRA Tagger."""

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "gpt2"
OUTPUT_DIR = "outputs/legal_lora"


# ==================================================
# MILESTONE 1: PREPARE LEGAL CLAUSE DATASET
# ==================================================

def milestone_1():
    """Load the LEDGAR legal clause dataset."""

    dataset = load_dataset("coastalcph/lex_glue", "ledgar")

    return {
        "train": dataset["train"],
        "validation": dataset["validation"],
        "test": dataset["test"],
        "num_labels": len(
            dataset["train"].features["label"].names
        ),
    }


# ==================================================
# MILESTONE 2: EXTEND GPT-2 TOKENIZER
# ==================================================

def milestone_2():
    """Add custom legal tags to GPT-2."""

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    model.resize_token_embeddings(len(tokenizer))

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token or legal_tags[0]

    if hasattr(model, "config"):
        model.config.pad_token_id = getattr(
            tokenizer, "pad_token_id", None
        )

    return {
        "tokenizer": tokenizer,
        "model": model,
        "legal_tags": legal_tags,
    }


# ==================================================
# MILESTONE 3: IMPLEMENT LoRA
# ==================================================

class LoRALayer(nn.Module):
    """Low-Rank Adaptation wrapper for a linear-like layer."""

    def __init__(self, base, rank=2, alpha=4, dropout=0.0):
        super().__init__()

        if rank <= 0:
            raise ValueError("rank must be positive")

        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        if hasattr(base, "in_features"):
            input_size = base.in_features
            output_size = base.out_features
        else:
            input_size = base.weight.shape[0]
            output_size = base.weight.shape[1]

        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.lora_A = nn.Parameter(
            torch.empty(rank, input_size)
        )

        self.lora_B = nn.Parameter(
            torch.zeros(output_size, rank)
        )

        nn.init.kaiming_uniform_(
            self.lora_A,
            a=5 ** 0.5
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
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
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    for parameter in model.parameters():
        parameter.requires_grad = False

    if not hasattr(model, "transformer"):
        raise ValueError("Expected a GPT-2 model.")

    for block in model.transformer.h:
        if not isinstance(block.attn.c_attn, LoRALayer):
            block.attn.c_attn = LoRALayer(
                block.attn.c_attn,
                rank=8,
                alpha=16,
                dropout=0.1
            )

    return model


# ==================================================
# MILESTONE 4: FINE-TUNE GPT-2 WITH LoRA
# ==================================================

def milestone_4(
    dataset=None,
    tokenizer=None,
    model=None,
    output_dir=OUTPUT_DIR,
    epochs=1,
    batch_size=4,
    learning_rate=2e-4,
    max_length=128,
):
    """Fine-tune GPT-2 using LoRA parameters."""

    if dataset is None:
        dataset = milestone_1()

    if tokenizer is None or model is None:
        loaded = milestone_2()

        if tokenizer is None:
            tokenizer = loaded["tokenizer"]

        if model is None:
            model = loaded["model"]

    has_lora = any(
        isinstance(module, LoRALayer)
        for module in model.modules()
    )

    if not has_lora and hasattr(model, "transformer"):
        model = milestone_3(model)

    trainable = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    if not trainable:
        raise ValueError("No trainable parameters found.")

    if epochs < 1 or batch_size < 1 or max_length < 2:
        raise ValueError("Invalid training configuration.")

    if getattr(tokenizer, "pad_token", None) is None:
        pad_token = getattr(tokenizer, "eos_token", None)
        if pad_token is not None:
            tokenizer.pad_token = pad_token

    if hasattr(model, "config"):
        model.config.pad_token_id = getattr(
            tokenizer, "pad_token_id", None
        )

    train_data = dataset["train"]

    label_names = None

    if hasattr(train_data, "features"):
        label_names = getattr(
            train_data.features["label"],
            "names",
            None
        )

    examples = []

    for row in train_data:
        clause = row["text"]
        label = row["label"]

        label_text = (
            label_names[int(label)]
            if label_names is not None
            else str(label)
        )

        examples.append(
            f"Legal clause: {clause}\n"
            f"Clause category: {label_text}"
        )

    if not examples:
        raise ValueError("Training dataset is empty.")

    device = next(model.parameters()).device

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        [
            p for p in model.parameters()
            if p.requires_grad
        ],
        lr=learning_rate
    )

    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0

        for start in range(0, len(examples), batch_size):

            batch = examples[start:start + batch_size]

            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        average_loss = epoch_loss / batch_count
        losses.append(average_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {average_loss:.4f}"
        )

    os.makedirs(output_dir, exist_ok=True)

    adapter_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    adapter_path = os.path.join(
        output_dir,
        "adapter.pt"
    )

    torch.save(adapter_state, adapter_path)

    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_dir)

    print("Training completed!")
    print("Training losses:", losses)
    print("Adapter saved at:", adapter_path)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "losses": losses,
        "adapter_path": adapter_path,
    }


# ==================================================
# MILESTONE 5: MERGE LoRA AND PERFORM INFERENCE
# ==================================================

def _merge_lora_layers(model):
    """Merge trained LoRA weights into their base layers."""

    def merge_module(module):
        for name, child in list(module.named_children()):

            if isinstance(child, LoRALayer):

                delta = (
                    child.lora_B @ child.lora_A
                ) * child.scaling

                with torch.no_grad():
                    if hasattr(child.base, "in_features"):
                        child.base.weight.add_(
                            delta.to(child.base.weight.dtype)
                        )
                    else:
                        child.base.weight.add_(
                            delta.T.to(child.base.weight.dtype)
                        )

                setattr(module, name, child.base)

            else:
                merge_module(child)

    merge_module(model)

    return model


def milestone_5(
    clause="This agreement shall be governed by the laws of California.",
    model=None,
    tokenizer=None,
    adapter_path=None,
    max_new_tokens=20,
):
    """Merge LoRA adapter and classify a new legal clause."""

    if model is None or tokenizer is None:
        loaded = milestone_2()

        if model is None:
            model = loaded["model"]

        if tokenizer is None:
            tokenizer = loaded["tokenizer"]

    has_lora = any(
        isinstance(module, LoRALayer)
        for module in model.modules()
    )

    if not has_lora and hasattr(model, "transformer"):
        model = milestone_3(model)

    if adapter_path is None:
        adapter_path = os.path.join(
            OUTPUT_DIR,
            "adapter.pt"
        )

    if os.path.exists(adapter_path):
        adapter_state = torch.load(
            adapter_path,
            map_location="cpu",
            weights_only=True
        )

        model.load_state_dict(
            adapter_state,
            strict=False
        )

    if has_lora or any(
        isinstance(module, LoRALayer)
        for module in model.modules()
    ):
        model = _merge_lora_layers(model)

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if hasattr(model, "config"):
        model.config.pad_token_id = tokenizer.pad_token_id

    device = next(model.parameters()).device
    model.to(device)
    model.eval()

    prompt = (
        f"Legal clause: {clause}\n"
        f"Clause category:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    new_tokens = generated[
        :, inputs["input_ids"].shape[1]:
    ]

    prediction = tokenizer.decode(
        new_tokens[0],
        skip_special_tokens=True
    ).strip()

    return {
        "clause": clause,
        "prediction": prediction,
        "tagged_clause": (
            f"{clause}\nClause category: {prediction}"
        ),
        "model": model,
        "tokenizer": tokenizer,
    }


# ==================================================
# MILESTONE 6: EVALUATE ACCURACY AND EFFICIENCY
# ==================================================

def milestone_6(
    model=None,
    tokenizer=None,
    dataset=None,
    max_samples=100,
    max_length=128,
):
    """Evaluate tagging accuracy and model efficiency."""

    if dataset is None:
        dataset = milestone_1()

    if model is None or tokenizer is None:
        loaded = milestone_2()

        if model is None:
            model = loaded["model"]

        if tokenizer is None:
            tokenizer = loaded["tokenizer"]

    test_data = dataset["test"]

    label_names = None

    if hasattr(test_data, "features"):
        label_names = getattr(
            test_data.features["label"],
            "names",
            None
        )

    if label_names is None:
        raise ValueError("Dataset must provide label names.")

    if max_samples < 1:
        raise ValueError("max_samples must be positive.")

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    if hasattr(model, "config"):
        model.config.pad_token_id = tokenizer.pad_token_id

    device = next(model.parameters()).device

    model.to(device)
    model.eval()

    correct = 0
    predictions = []
    true_labels = []

    total = min(len(test_data), max_samples)

    start_time = time.perf_counter()

    with torch.inference_mode():

        for i in range(total):

            row = test_data[i]

            clause = row["text"]
            true_label = label_names[int(row["label"])]

            prompt = (
                f"Legal clause: {clause}\n"
                f"Clause category:"
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            generated = model.generate(
                **inputs,
                max_new_tokens=12,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

            new_tokens = generated[
                :, inputs["input_ids"].shape[1]:
            ]

            predicted_text = tokenizer.decode(
                new_tokens[0],
                skip_special_tokens=True
            ).strip().lower()

            predicted_label = next(
                (
                    label for label in label_names
                    if label.lower() in predicted_text
                ),
                ""
            )

            if predicted_label.lower() == true_label.lower():
                correct += 1

            predictions.append(predicted_label)
            true_labels.append(true_label)

    elapsed = time.perf_counter() - start_time

    accuracy = correct / total if total else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "inference_time": elapsed,
        "samples_per_second": (
            total / elapsed if elapsed > 0 else 0.0
        ),
        "predictions": predictions,
        "true_labels": true_labels,
    }
