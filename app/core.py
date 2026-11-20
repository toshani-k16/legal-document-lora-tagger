
"""Core implementation for the Legal Document LoRA Tagger."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==================================================
# MILESTONE 1: Prepare legal clause dataset
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
# MILESTONE 2: Extend GPT-2 tokenizer and model
# ==================================================

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


# ==================================================
# MILESTONE 3: Implement LoRA adapter
# ==================================================

class LoRALayer(nn.Module):
    """Low-Rank Adaptation wrapper."""

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
    """Inject LoRA adapters into GPT-2."""

    if model is None:
        model = AutoModelForCausalLM.from_pretrained("gpt2")

    for parameter in model.parameters():
        parameter.requires_grad = False

    for block in model.transformer.h:

        block.attn.c_attn = LoRALayer(
            block.attn.c_attn,
            rank=8,
            alpha=16,
            dropout=0.1
        )

    return model


# ==================================================
# MILESTONE 4: Fine-tune GPT-2 with LoRA
# ==================================================

def milestone_4(
    dataset=None,
    tokenizer=None,
    model=None,
    output_dir="outputs/legal_lora",
    epochs=1,
    batch_size=4,
    learning_rate=2e-4,
    max_length=128,
):

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

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise ValueError("No trainable LoRA parameters found.")

    if epochs < 1 or batch_size < 1 or max_length < 2:
        raise ValueError("Invalid training configuration.")

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = (
            getattr(tokenizer, "eos_token", None)
            or "<|pad|>"
        )

    if hasattr(model, "config"):
        model.config.pad_token_id = getattr(
            tokenizer,
            "pad_token_id",
            None
        )

    train_data = dataset["train"]

    label_names = None

    if hasattr(train_data, "features"):
        label_feature = train_data.features["label"]
        label_names = getattr(label_feature, "names", None)

    examples = []

    for row in train_data:

        clause = row["text"]
        label = row["label"]

        if label_names is not None:
            label_text = label_names[int(label)]
        else:
            label_text = str(label)

        examples.append(
            f"Legal clause: {clause}\n"
            f"Clause category: {label_text}"
        )

    if not examples:
        raise ValueError("Training dataset is empty.")

    device = next(model.parameters()).device

    model.to(device)
    model.train()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate
    )

    losses = []

    for epoch in range(epochs):

        epoch_loss = 0.0
        batch_count = 0

        for start in range(0, len(examples), batch_size):

            batch_texts = examples[
                start:start + batch_size
            ]

            encoded = tokenizer(
                batch_texts,
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

    return {
        "model": model,
        "tokenizer": tokenizer,
        "losses": losses,
        "adapter_path": adapter_path,
    }


# ==================================================
# MILESTONE 5: Merge LoRA and perform inference
# ==================================================

def merge_lora_layer(layer):
    """Merge trained LoRA weights into the original layer."""

    if not isinstance(layer, LoRALayer):
        raise TypeError("Expected a LoRALayer.")

    base = layer.base

    with torch.no_grad():

        if hasattr(base, "in_features"):
            # Standard PyTorch Linear layer.
            delta = layer.lora_B @ layer.lora_A

        else:
            # Hugging Face GPT-2 Conv1D layer.
            delta = layer.lora_A.T @ layer.lora_B.T

        base.weight.add_(
            delta.to(
                device=base.weight.device,
                dtype=base.weight.dtype
            ),
            alpha=layer.scaling
        )

    return base


def merge_lora_model(model):
    """Merge every LoRA adapter in the GPT-2 model."""

    merged_count = 0

    for block in model.transformer.h:

        layer = block.attn.c_attn

        if isinstance(layer, LoRALayer):

            block.attn.c_attn = merge_lora_layer(layer)
            merged_count += 1

    if merged_count == 0:
        raise ValueError("No LoRA adapters found to merge.")

    return model


def milestone_5(
    clause,
    model=None,
    tokenizer=None,
    adapter_path="outputs/legal_lora/adapter.pt",
    model_name="gpt2",
    max_new_tokens=20,
):
    """Load trained LoRA weights, merge them and tag a new clause."""

    if not clause or not clause.strip():
        raise ValueError("Please provide a legal clause.")

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(
            f"Adapter not found: {adapter_path}"
        )

    # Load tokenizer.
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.dirname(adapter_path)
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load GPT-2.
    if model is None:

        model = AutoModelForCausalLM.from_pretrained(
            model_name
        )

        model.resize_token_embeddings(len(tokenizer))

        model = milestone_3(model)

    # Load trained adapter parameters.
    adapter_state = torch.load(
        adapter_path,
        map_location="cpu",
        weights_only=True
    )

    incompatible = model.load_state_dict(
        adapter_state,
        strict=False
    )

    if incompatible.unexpected_keys:
        raise ValueError(
            f"Unexpected adapter keys: {incompatible.unexpected_keys}"
        )

    loaded_count = sum(
        key.endswith(("lora_A", "lora_B"))
        for key in adapter_state
    )

    if loaded_count == 0:
        raise ValueError("No LoRA weights found in adapter.")

    # Merge trained weights into GPT-2.
    model = merge_lora_model(model)

    model.eval()

    device = next(model.parameters()).device

    # Prepare new legal clause.
    prompt = (
        f"Legal clause: {clause}\n"
        "Clause category:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Generate predicted category.
    with torch.no_grad():

        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    prompt_length = inputs["input_ids"].shape[1]

    new_tokens = generated[0][prompt_length:]

    prediction = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    return {
        "clause": clause,
        "prediction": prediction,
        "tagged_document": (
            f"Legal clause: {clause}\n"
            f"Clause category: {prediction}"
        ),
        "model": model,
        "tokenizer": tokenizer,
    }


# ==================================================
# MILESTONE 6: Evaluate tagging accuracy
# ==================================================

def milestone_6():
    raise NotImplementedError("TODO(milestone 6)")
