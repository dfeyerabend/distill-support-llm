# Distil Support LLM — German Customer Support via Knowledge Distillation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![HuggingFace – Teacher](https://img.shields.io/badge/HuggingFace-Teacher%20Adapter-orange)](https://huggingface.co/Feyerade/german-support-qwen-lora-adapter)
[![HuggingFace – Student](https://img.shields.io/badge/HuggingFace-Student%20Model-orange)](https://huggingface.co/Feyerade/german-support-student-1.5b-distilled)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

> **Showcase project.** Built as a portfolio piece demonstrating QLoRA fine-tuning and knowledge distillation on a domain-specific task. The pipeline is end-to-end reproducible via the notebooks and pre-trained models on Hugging Face.
- To-DO host model on railway
---

## Overview

This project fine-tunes a 3B-parameter LLM on a minimal synthetic dataset of 33 German customer support examples using **QLoRA** — intentionally scoped for style and format adaptation rather than factual knowledge transfer. 
That domain style is then distilled into a 1.5B student model via **sequence-level knowledge distillation** (Kim & Rush, 2016).

The result: a smaller, faster model that retains the teacher's domain behaviour — deployable on hardware where the teacher model would not fit.

**Pipeline in three steps:**

1. **Teacher fine-tuning** — `Qwen2.5-3B-Instruct` + QLoRA on 33 German support Q&A pairs
2. **Data generation** — the teacher generates responses for 234 diverse support prompts, creating the distillation dataset
3. **Student training** — `Qwen2.5-1.5B-Instruct` trained on teacher-generated (prompt, response) pairs

---

## Models

| Model | HuggingFace | Parameters | Role |
|---|---|---|---|
| Teacher (LoRA Adapter) | [Feyerade/german-support-qwen-lora-adapter](https://huggingface.co/Feyerade/german-support-qwen-lora-adapter) | 3B (base) | Fine-tuned domain expert |
| Student | [Feyerade/german-support-student-1.5b-distilled](https://huggingface.co/Feyerade/german-support-student-1.5b-distilled) | 1.5B | Distilled, deployable model |

Base model: `Qwen2.5-Instruct` (Qwen Team, Apache 2.0).

---

## Results

*Full evaluation — BERTScore, latency, and VRAM benchmarks — is in [`results/teacher_vs_student.md`](results/teacher_vs_student.md) and Notebook 4.*

| Metric | Teacher (3B) | Student (1.5B) |
|---|---|---|
| BERTScore F1 | — | — |
| Avg. latency (T4) | — | — |
| VRAM (inference) | — | — |
| Model size on disk | — | — |

---

## Repo Structure

```
distil-support-llm/
├── notebooks/
│   ├── 01_teacher_finetuning.ipynb     # QLoRA fine-tuning of the teacher model
│   ├── 02_data_generation.ipynb        # Teacher generates the distillation dataset
│   ├── 03_student_distillation.ipynb   # Student training on teacher outputs
│   └── 04_evaluation.ipynb             # Teacher vs. Student: quality, latency, VRAM
├── src/
│   ├── data/                           # Dataset loading and prompt formatting
│   ├── training/                       # Training loop helpers and configs
│   ├── distillation/                   # Teacher generation, student training
│   └── evaluation/                     # BERTScore, latency, VRAM measurement
├── demo/                               # Gradio app (Hugging Face Space)
├── configs/                            # Hyperparameter configs (YAML)
├── results/                            # Evaluation outputs, plots, comparison tables
└── requirements.txt
```

---

## Getting Started

### Hardware requirements

Training notebooks require a CUDA-capable GPU with at least **16 GB VRAM** (developed on a Kaggle T4).

> **Windows users:** `unsloth` and `bitsandbytes` have limited native Windows support due to CUDA build dependencies. Running the training notebooks on Windows requires **WSL2** or a cloud environment (Kaggle, Colab). The demo and evaluation notebooks run fine natively.

### Installation

```bash
git clone https://github.com/Feyerade/distil-support-llm.git
cd distil-support-llm
pip install -r requirements.txt
```

> `torch` must be installed separately with the correct CUDA version for your system.
> See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the right command.

### Environment variables

```bash
export HF_TOKEN=your_huggingface_token   # required for pushing models to the Hub
```

On Windows (PowerShell): `$env:HF_TOKEN = "your_token"`

---

## Notebooks

Each notebook is self-contained and runs sequentially. Notebooks 1–3 require a GPU; Notebook 4 can run on CPU.

| Notebook | Description | GPU required |
|---|---|---|
| `01_teacher_finetuning` | Fine-tunes the teacher model with QLoRA | Yes |
| `02_data_generation` | Generates distillation dataset via teacher inference | Yes |
| `03_student_distillation` | Trains the student on teacher-generated data | Yes |
| `04_evaluation` | Benchmarks both models | No |

All training notebooks were developed and run on **Kaggle** (T4 GPU, 16 GB VRAM).
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](YOUR_KAGGLE_LINK)

---

## Approach

### Fine-Tuning (QLoRA)

The teacher (`Qwen2.5-3B-Instruct`) is fine-tuned with a QLoRA: the base model weights are quantized to 4-bit, and small trainable LoRA adapters (`r=16`) are attached to all attention and MLP projections.   
Only the adapter weights are updated — ~1% of total parameters — making training feasible on a single consumer GPU.

Training dataset: 33 synthetic German customer support Q&A pairs across categories including payments, shipping, returns, account security, and technical issues.

### Knowledge Distillation (Sequence-Level)

This project uses **sequence-level / response-based distillation** (Kim & Rush, 2016), not classical logit-level KD (Hinton, 2015).

Concretely: the fine-tuned teacher generates responses for 234 German support prompts. The student is then trained via standard SFT on these (prompt, teacher-response) pairs. The student never sees the teacher's internal probability distributions — only its generated outputs.

This is a deliberate scope decision. Logit-level distillation (matching soft probability distributions via temperature-scaled KL divergence) would require both models in memory simultaneously and is left as a potential extension.

---

## Limitations

- **Dataset size:** The fine-tuning dataset is intentionally minimal (33 examples) — sufficient for style and format adaptation (response tone, language, structure), but not for teaching factual domain knowledge.
- **Synthetic data only:** All training data is synthetically generated. No real customer interactions were used.
- **Distillation method:** Sequence-level KD does not transfer the teacher's uncertainty or soft label information. Quality ceiling is the teacher's generation quality.
- **Hardware dependency:** Training requires a CUDA GPU; not reproducible on CPU.

---

## References

- Kim & Rush (2016) — [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947)
- Hinton et al. (2015) — [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Unsloth](https://github.com/unslothai/unsloth)
- [Qwen2.5 on Hugging Face](https://huggingface.co/Qwen)

---

## Author

**Dennis Feyerabend** · May 2026  
[github.com/Feyerade](https://github.com/Feyerade) · [huggingface.co/Feyerade](https://huggingface.co/Feyerade)

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.  
Base model weights: `Qwen2.5-Instruct` (Apache 2.0, Qwen Team).
