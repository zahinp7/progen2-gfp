# ProGen2 Fine-tuning for Green Fluorescent Protein Design

Fine-tuned [ProGen2](https://github.com/salesforce/progen) — a large language model for protein sequences — on the green fluorescent protein (GFP) family to generate novel GFP variants with high predicted structural fidelity. Generated sequences were evaluated using AlphaFold3, with top candidates achieving **pLDDT > 92** and **pTM = 0.92**.

---

## Overview

Protein language models (pLMs) learn evolutionary patterns across millions of protein sequences, making them powerful tools for conditional protein design. This project adapts ProGen2-small to the GFP family through supervised fine-tuning, then uses the fine-tuned model to autoregressively generate new sequence variants conditioned on a 64-token GFP prefix.

The pipeline covers:
- Fine-tuning a pretrained pLM on a curated GFP dataset
- Autoregressive sequence generation using greedy and top-p (nucleus) decoding
- Structure prediction and candidate ranking via AlphaFold3 confidence metrics (pLDDT, pTM)

---

## Model Architecture

The model is `ProGen2-small`, a decoder-only transformer adapted for protein sequences:

| Component | Detail |
|---|---|
| Architecture | Decoder-only transformer (GPT-style) |
| Embedding dimension | 4096 |
| Layers | 28 |
| Attention heads | 16 |
| Position encoding | Rotary positional embeddings (RoPE) |
| Vocabulary | 50,400 tokens (amino acid alphabet + special tokens) |
| Max sequence length | 2048 |

The `forward()` method maps hidden states through `lm_head` and applies softmax to produce per-token probability distributions over the vocabulary. At inference, `forward_inference()` autoregressively extends the prefix sequence until an EOS token is predicted or `MAX_SEQ_LENGTH` (512) is reached.

---

## Training

Fine-tuning was performed on PSC Bridges2 GPU infrastructure.

**Hyperparameters (fixed):**

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate (init) | 1e-3 |
| LR schedule | Inverse square root (`lr = decay_factor × updates^-0.5`) |
| LR scheduler updates | 4000 |
| Epochs | 10 |
| Batch size | 32 |

**Loss function:** Masked cross-entropy over non-padding positions:

```
loss = mean( -log P(target_token) * (target != pad_idx) )
```

The best checkpoint (lowest validation loss) was saved to `models/best_checkpoint.pt`.

---

## Inference / Sequence Design

The fine-tuned model generates new GFP variants by taking the first 64 tokens of a test sequence as a prompt and completing it autoregressively.

**Two decoding strategies are supported:**

- **Greedy** — always selects the highest-probability token at each step (deterministic)
- **Top-p (nucleus sampling, p=0.8)** — samples from the smallest set of tokens whose cumulative probability exceeds 0.8 (stochastic, more diverse outputs)

92 sequence designs were generated and written to `design/design.txt`.

---

## AlphaFold3 Evaluation

Each generated sequence was submitted to the [AlphaFold3 web server](https://alphafoldserver.com) for structure prediction. Candidates were ranked by **pLDDT** (per-residue confidence) with a minimum **pTM > 0.8** filter for global fold confidence.

### Top-5 Candidates

| Rank | Design | pLDDT | pTM |
|---|---|---|---|
| 1 | fold_design22 | 92.632 | 0.92 |
| 2 | fold_design92 | 92.455 | 0.92 |
| 3 | fold_design56 | 92.445 | 0.92 |
| 4 | fold_design18 | 92.336 | 0.92 |
| 5 | fold_design74 | 92.204 | 0.92 |

All top candidates exceeded the pTM > 0.8 threshold by a wide margin, indicating that fine-tuning successfully constrained the model to the GFP structural domain. Predicted structures showed intact β-barrel geometry consistent with canonical GFP topology.

> Full candidate report with AlphaFold3 visualizations, sequences, and metric explanations is available in [`report/GFP_Design_Report.pdf`](report/GFP_Design_Report.pdf).

---

## Repository Structure

```
progen2-gfp/
├── model.py              # ProGen2 model architecture (transformer + causal LM head)
├── train.py              # Fine-tuning script
├── inference.py          # Sequence generation script
├── report/
│   └── GFP_Design_Report.pdf   # Top-5 candidate report with AF3 results
└── README.md
```

> **Note:** Pretrained model weights, training data, and generated sequences are not included in this repository. See Setup below for how to reproduce.

---

## Setup & Reproduction

```bash
# Create environment
conda create -n progen python=3.9.16 -y
conda activate progen
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.49.0

# Download pretrained weights
mkdir -p pretrained_model/progen2-small models design
cd pretrained_model/progen2-small
wget https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-small.tar.gz
tar -xvzf progen2-small.tar.gz
cd ../..

# Fine-tune
python train.py

# Generate sequences
python inference.py
```

---

## Skills Demonstrated

- Fine-tuning large pretrained protein language models (ProGen2) on domain-specific data
- Implementing inverse square root learning rate scheduling from scratch
- Autoregressive sequence generation with greedy and nucleus (top-p) decoding
- Structural evaluation of designed proteins using AlphaFold3 confidence metrics
- Running GPU workloads on HPC infrastructure (PSC Bridges2)

---

## References

- Nijkamp et al. (2023). [ProGen2: Exploring the Space of Protein Language Models](https://www.cell.com/cell-systems/fulltext/S2405-4712(23)00272-7). *Cell Systems.*
- Abramson et al. (2024). [Accurate structure prediction of biomolecular interactions with AlphaFold3](https://www.nature.com/articles/s41586-024-07487-w). *Nature.*

---
