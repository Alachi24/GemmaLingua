# GemmaLingua

**Fine-tuning Gemma 4 for Igbo — a low-resource African language.**

GemmaLingua adapts Google's [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it) instruction-tuned model to understand and generate [Igbo](https://en.wikipedia.org/wiki/Igbo_language), a language spoken by over 45 million people in Nigeria with very limited representation in existing AI training data.

The project fine-tunes the base model across three chained stages — language modelling, translation, and span infilling — each building on the previous adapter. The pipeline is designed to be retargeted at other low-resource Nigerian languages (Yoruba, Hausa, and others) by changing four fields in a single config file.

---

## Table of Contents

- [Why Igbo](#why-igbo)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Adapting to Another Language](#adapting-to-another-language)
- [Training Results](#training-results)
- [Contributing](#contributing)

---

## Why Igbo

Most large language models perform poorly on Igbo because the language is severely under-represented in public training corpora. Benchmark evaluations (IrokoBench, AfroBench) consistently show a large performance gap between high-resource languages like English and French versus low-resource African languages.

GemmaLingua is a focused effort to close part of that gap — starting with Igbo — by fine-tuning a capable base model on real Igbo text data rather than relying on incidental coverage in general pretraining.

---

## How It Works

Training happens in three sequential stages. Each stage produces a LoRA adapter that is loaded as the starting point for the next stage, so capabilities accumulate rather than overwrite each other.

```
google/gemma-4-E4B-it  (base)
        │
        ▼  Stage 1 — Language Modelling
igbo-gemma4-lm-finetuned
        │
        ▼  Stage 2 — English → Igbo Translation
igbo-gemma4-finetuned
        │
        ▼  Stage 3 — Span Infilling (fill-in-the-blank)
igbo-gemma4-masking-finetuned  ← final model
```

### Stage 1 — Language Modelling
Trains on raw Igbo text (`.txt` or `.csv` files) using a standard causal language modelling objective. The model learns general Igbo vocabulary, grammar, and sentence structure.

### Stage 2 — Translation
Trains on English–Igbo sentence pairs using the prompt format:
```
Translate English to Igbo.
English: {src}
Igbo: {tgt}
```
Chains on top of the Stage 1 adapter.

### Stage 3 — Span Infilling
Trains on Igbo sentences with randomly blanked spans using the prompt format:
```
Fill in the blanks in the Igbo sentence.
Igbo: {masked}
Answer: {answer}
```
Chains on top of the Stage 2 adapter. This is the final model.

All stages use **LoRA** (Low-Rank Adaptation) with 4-bit quantization, making training feasible on a single GPU.

---

## Project Structure

```
GemmaLingua/
├── config/
│   └── pipeline.yaml          # Single config file controlling all stages
├── python/
│   ├── cache_model.py         # Download and cache the base model locally
│   ├── train_lm.py            # Stage 1: language model fine-tuning
│   ├── train_translation.py   # Stage 2: translation fine-tuning
│   ├── train_masking.py       # Stage 3: span infilling fine-tuning
│   ├── eval_lm.py             # Evaluate language model (perplexity, token accuracy)
│   ├── eval_translation.py    # Evaluate translation (chrF, BLEU)
│   ├── eval_masking.py        # Evaluate span infilling (exact match, token accuracy)
│   ├── download_translation.py # Download public Igbo translation datasets
│   ├── requirements.txt       # Python dependencies
│   └── utils/
│       ├── data.py            # Dataset loading utilities
│       └── model.py           # Model loading, LoRA config, adapter chaining
├── data/
│   └── raw/
│       ├── monolingual/       # Raw Igbo text files for Stage 1
│       └── igbo_translation_train.csv  # English–Igbo pairs for Stages 2 & 3
├── models/
│   ├── base/                  # Cached base model (auto-created)
│   └── adapters/              # Trained LoRA adapters (auto-created)
│       ├── igbo-gemma4-lm-finetuned/
│       ├── igbo-gemma4-finetuned/
│       └── igbo-gemma4-masking-finetuned/
├── scripts/                   # Shell scripts for running the pipeline
├── EVAL_PLAN.md               # Internal evaluation planning document
└── pom.xml                    # Maven wrapper (pipeline orchestration)
```

---

## Requirements

### Hardware
- GPU with at least **16 GB VRAM** recommended (tested on A100/V100)
- 4-bit quantization is used by default — 24 GB+ recommended for comfortable training
- CPU-only is not supported for training

### Software
- Python 3.10 or higher
- CUDA 11.8 or higher
- A [Hugging Face](https://huggingface.co) account with access to `google/gemma-4-E4B-it`

---

## Setup

**1. Clone the repository:**
```bash
git clone https://github.com/Mark12323/GemmaLingua.git
cd GemmaLingua
```

**2. Create a virtual environment and install dependencies:**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r python/requirements.txt
```

**3. Set your Hugging Face token:**

You need access to the Gemma 4 model on Hugging Face. Request access at
https://huggingface.co/google/gemma-4-E4B-it then set your token:

```bash
export HF_TOKEN=your_token_here   # Windows: set HF_TOKEN=your_token_here
```

**4. Cache the base model locally** (recommended — avoids re-downloading during training):
```bash
python python/cache_model.py \
  --model-id google/gemma-4-E4B-it \
  --cache-dir ./models/base
```

---

## Data Preparation

### Stage 1 — Monolingual Igbo Text

Place any number of `.txt` or `.csv` files containing Igbo text into:
```
data/raw/monolingual/
```

All files in the directory are picked up automatically. There is no required format — plain Igbo sentences, one per line, work well.

### Stages 2 & 3 — Translation Data

Provide a CSV file with at least two columns — `en` (English) and `ig` (Igbo):

```csv
en,ig
"Good morning","Ụtụtụ ọma"
"How are you?","Kedu ka ị mere?"
```

Place the file at `data/raw/igbo_translation_train.csv` or update the path in `config/pipeline.yaml`.

To download a public English–Igbo dataset:
```bash
python python/download_translation.py
```

> **Important:** Set aside a held-out slice (last ~10% of rows) of your data **before** training. This held-out set is required for valid evaluation. Do not include held-out examples in any training file.

---

## Training

All training hyperparameters are controlled by `config/pipeline.yaml`. Edit that file to change learning rate, batch size, LoRA rank, number of epochs, and data paths before running any training script.

### Stage 1 — Language Model
```bash
python python/train_lm.py \
  --data data/raw/monolingual/ \
  --output ./models/adapters/igbo-gemma4-lm-finetuned
```

### Stage 2 — Translation
```bash
python python/train_translation.py \
  --data data/raw/igbo_translation_train.csv \
  --adapter ./models/adapters/igbo-gemma4-lm-finetuned \
  --output ./models/adapters/igbo-gemma4-finetuned
```

### Stage 3 — Span Infilling
```bash
python python/train_masking.py \
  --data data/raw/igbo_translation_train.csv \
  --adapter ./models/adapters/igbo-gemma4-finetuned \
  --output ./models/adapters/igbo-gemma4-masking-finetuned
```

Each stage saves its adapter to `./models/adapters/`. Training times on a single A100:

| Stage | Approximate Runtime |
|---|---|
| Language Model | ~14.5 hours |
| Translation | ~2.5 hours |
| Span Infilling | ~2.3 hours |

---

## Evaluation

Run evaluation after training using the held-out data slice. Each script compares the base model against the fine-tuned adapter on the same examples.

### Language Model — Perplexity and Token Accuracy
```bash
python python/eval_lm.py \
  --data data/raw/monolingual/held_out.txt \
  --adapter ./models/adapters/igbo-gemma4-lm-finetuned
```

Reports: perplexity (`e^loss`) and top-1 next-token accuracy on held-out Igbo text.

### Translation — chrF and BLEU
```bash
python python/eval_translation.py \
  --data data/raw/igbo_translation_train_held_out.csv \
  --adapter ./models/adapters/igbo-gemma4-finetuned
```

Reports: chrF / chrF++ (primary — character-level, robust for morphologically rich Igbo) and BLEU (secondary — conventional MT metric).

### Span Infilling — Exact Match and Token Accuracy
```bash
python python/eval_masking.py \
  --data data/raw/igbo_translation_train_held_out.csv \
  --adapter ./models/adapters/igbo-gemma4-masking-finetuned
```

Reports: span exact-match percentage and token accuracy on blanked spans only (copied context is not scored).

> **Fairness note:** The base model is evaluated with 3–5 few-shot in-format examples; the fine-tuned adapter runs zero-shot. The same prompt format used during training is used for evaluation. Both models are loaded with 4-bit quantization.

---

## Adapting to Another Language

The pipeline is designed to be retargeted at any language with parallel text data. Change these four fields in `config/pipeline.yaml` — nothing else needs to change:

```yaml
language:
  source_code: "en"        # CSV column for source text
  source_name: "English"   # Name used in the prompt
  target_code: "yo"        # CSV column for target text  ← change this
  target_name: "Yoruba"    # Name used in the prompt     ← and this
```

The prompt templates, data loaders, and training scripts all read from these fields automatically.

---

## Training Results

Results from the initial training run on Igbo:

| Stage | Adapter | Train Loss | Token Accuracy | Runtime |
|---|---|---|---|---|
| Language Model | `igbo-gemma4-lm-finetuned` | 2.105 (recent: ~1.97) | **59.9%** | ~14.5 h |
| Translation | `igbo-gemma4-finetuned` | 1.42 (recent: ~1.05) | **77.1%** | ~2.4 h |
| Span Infilling | `igbo-gemma4-masking-finetuned` | 1.49 (recent: ~1.13) | **75.9%** | ~2.3 h |

**On the 59.9% language model token accuracy:** This is top-1 next-token accuracy on raw Igbo text. The healthy range for free-text causal language modelling is 50–70%...59.9% on a low-resource language with limited training data is a solid result. Loss values are not comparable across stages because each stage optimises a different objective over a different token distribution.

A full base-vs-fine-tuned comparison using held-out data is in progress. See `EVAL_PLAN.md` for the detailed evaluation methodology.

---

## Contributing

Contributions are welcome, especially:

- **Igbo text data** — monolingual corpora, parallel translation pairs, or pointers to public datasets
- **Evaluation results** — running the eval scripts and sharing your numbers
- **Retargeting to other languages** — Yoruba, Hausa, Ibibio, and other Nigerian languages
- **Documentation improvements**

Please open an issue before starting significant work so efforts are not duplicated.

All documentation PRs should target the `staging` branch. See the issue tracker for open tasks.
