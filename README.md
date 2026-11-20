# Legal Document LoRA Tagger

A research prototype that uses GPT-2 with Low-Rank Adaptation (LoRA) to generate legal clause category tags.

## Overview

Legal contracts contain clauses such as governing law, indemnity, confidentiality, termination and liability. This project explores using parameter-efficient fine-tuning to help classify legal clause text.

## Technologies

- Python 3.12
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- GPT-2
- LoRA (Low-Rank Adaptation)
- LEDGAR dataset through LexGLUE
- pytest

## Project structure

```text
legal-document-lora-tagger/
├── app/
│   ├── __init__.py
│   ├── core.py
│   ├── cli.py
│   ├── llm.py
│   └── storage.py
├── tests/
│   ├── test_core.py
│   └── test_smoke.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── requirements.txt
├── pytest.ini
└── README.md
```

## Setup

Python 3.12 is recommended.

Clone the repository:

```bash
git clone https://github.com/toshani-k16/legal-document-lora-tagger.git
cd legal-document-lora-tagger
```

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The project uses the LEDGAR dataset available through the LexGLUE benchmark on Hugging Face.

The dataset contains legal provisions with clause-category labels. The dataset is loaded through the Hugging Face Datasets library.

The first run may download model and dataset files.

## Implementation milestones

1. Dataset preparation using LEDGAR.
2. GPT-2 tokenizer extension with legal clause tags.
3. LoRA layer implementation and adapter injection.
4. Fine-tuning GPT-2 using trainable LoRA parameters.
5. LoRA merging and clause-tag generation.
6. Evaluation of generated predictions.

Core implementations are located in `app/core.py`.

## Running tests

Run the test suite:

```bash
python -m pytest -q
```

Run the core tests:

```bash
python -m pytest -q tests/test_core.py
```

## Training and inference

The milestone functions are implemented in `app/core.py`.

- `milestone_1()` prepares the dataset.
- `milestone_2()` loads GPT-2 and extends its tokenizer.
- `milestone_3()` injects LoRA layers.
- `milestone_4()` performs LoRA fine-tuning and saves the adapter.
- `milestone_5()` generates a category for a supplied legal clause.
- `milestone_6()` evaluates generated predictions.

Refer to the function signatures and parameters in `app/core.py` when calling each function.

## Training experiment

An initial experimental run used 32 training examples for one epoch, with a batch size of 2 and a maximum sequence length of 96.

The reported training loss was approximately 3.6977.

This is a small proof-of-concept run, not evidence of production-level classification accuracy.

## Limitations

- The initial training experiment used a small subset of the dataset.
- More training data and multiple epochs are needed for reliable results.
- Generated labels may not always match the correct legal category.
- Evaluation accuracy must be measured on a held-out dataset before practical deployment.
- The prototype is not a substitute for legal review or professional legal advice.

## Reproducibility

Dependencies are listed in `requirements.txt`. Dataset and pretrained model downloads require an internet connection.

Training and inference may require substantial memory and processing time depending on the machine.

## Security

Do not commit `.env`, virtual environments, API keys, credentials or other secrets.

## Author

Toshani Kanive
