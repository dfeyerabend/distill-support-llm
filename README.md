# Distil Support LLM — German Customer Support via Knowledge Distillation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![HuggingFace – Teacher](https://img.shields.io/badge/HuggingFace-Teacher%20Adapter-orange)](https://huggingface.co/Feyerade/german-support-leollm-lora-adapter)
[![HuggingFace – Student](https://img.shields.io/badge/HuggingFace-Student%20Model-orange)](https://huggingface.co/Feyerade/german-support-llama-1b-distilled)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

> **Showcase project.** Built as a portfolio piece demonstrating QLoRA fine-tuning and knowledge distillation on a domain-specific task. The pipeline is end-to-end reproducible via the notebooks and pre-trained models on Hugging Face.

---

## Overview

This project fine-tunes a 7B-parameter LLM on a minimal synthetic dataset of 28 German customer support examples using **QLoRA** — intentionally scoped for style and format adaptation rather than factual knowledge transfer. 
That domain style is then distilled into a 1B student model via **sequence-level knowledge distillation** (Kim & Rush, 2016).

The result: a smaller, faster model that retains the teacher's domain behaviour — deployable on hardware where the teacher model would not fit.

**Pipeline in three steps:**

1. **Teacher fine-tuning** — `LeoLM/leo-mistral-hessianai-7b-chat` + QLoRA on 28 German support Q&A pairs
2. **Data generation** — the teacher generates responses for 232 diverse support prompts, creating the distillation dataset
3. **Student training** — `Llama-3.2-1B-Instruct` trained on teacher-generated (prompt, response) pairs

---

## Models

| Model | HuggingFace | Parameters | Role |
|---|---|---|---|
| Teacher (LoRA Adapter) | [Feyerade/german-support-leollm-lora-adapter](https://huggingface.co/Feyerade/german-support-leollm-lora-adapter) | 7B (base) | Fine-tuned domain expert |
| Student | [Feyerade/german-support-llama-1b-distilled](https://huggingface.co/Feyerade/german-support-llama-1b-distilled) | 1B | Distilled, deployable model |

Teacher base: `LeoLM/leo-mistral-hessianai-7b-chat` (HessianAI, Apache 2.0).  
Student base: `meta-llama/Llama-3.2-1B-Instruct` (Meta, Llama 3.2 Community License).

---

## Results

- Evaluation runs all three models on 60 German customer-support queries and scores each response with Claude Haiku as a blind LLM judge (4 binary criteria).
- The queries were scored on the the presence of: The presence of a leading sentence (acknowledgement), the presence of clear, numbered list of steps, the presence of a closing sentence, and overall professional tone.
- Full methodology in `notebooks/05-evaluation.ipynb`.

| Metric                 | Base (1B) | Teacher (7B + LoRA) | Student (1B distilled) |
|------------------------|---|---|---|
| Avg Format Score (0–4) | 2.57 | 3.47 | 3.30 |
| Acknowledgement        | 88.3% | 71.7% | 93.3% |
| Structured Steps       | 60.0% | 88.3% | 73.3% |
| Closing Sentence       | 53.3% | 96.7% | 95.0% |
| Professional Tone      | 55.0% | 90.0% | 68.3% |
| Avg Word Count         | 99.9 | 68.5 | 69.0 |
| Avg Tokens/sec (T4)    | 43.2 | 16.0 | 43.2 |
| Peak VRAM (4-bit)      | 1.14 GB | 4.67 GB | 1.14 GB |

### **Key findings**
**Overall Scores**
- The student (3.30) and teacher (3.47) both score substantially above the untrained base (2.57) on the 0–4 format scale.

**Acknowledgement** 
- The score is highest in the student (93.3%), even above the teacher (71.7%). 
- The base Llama 3.2 1B already opens empathetically by default (88.3%) — the teacher's stricter format prompt appears to suppress this (71.7%), as the model prioritises jumping into the numbered steps. The student retains the base's natural opener while also absorbing the teacher's structure, combining the best of both (93.3%).

**Closing sentence** 
- A closing sentence was included very reliably in student responses (95.0%) and teacher responses (96.7%), while being far less common in base responses (only present in 53.3%). 
- This is the largest single gain across all criteria from base to student (+41.7 percentage points).

**Structured steps** 
- Transfer is partial (73.3% vs 88.3% teacher) — the expected trade-off at a 7:1 parameter ratio. Still a meaningful gain over the 60.0% base.

**Professional Tone**
- The student (68.3%) shows a clear gain over the base (55.0%), but falls furthest from the teacher (90.0%) on this criterion.
- Tone is a subtle, distributed property — harder to absorb from 232 training examples than structural features like closing sentences or numbered steps.

**Speed and VRAM match the base model**
- The student runs at 43.2 tok/s on 1.14 GB VRAM — 2.7× faster and 75% lighter than the 7B teacher — while remaining within 0.17 format points of the teacher.

---

## Repo Structure

```
distil-support-llm/
├── notebooks/
│   ├── 01-teacher-finetuning.ipynb     # QLoRA fine-tuning of the teacher model
│   ├── 02-data-generation.ipynb        # Teacher generates the distillation dataset
│   ├── 03-student-distillation.ipynb   # Student training on teacher outputs
│   ├── 04-inference.ipynb              # Runs all three models on 60 evaluation queries
│   ├── 05-evaluation.ipynb             # LLM judge + analysis + summary table (local)
│   └── 05-evaluation-kaggle.ipynb      # Kaggle version — loads data from HuggingFace Hub
├── results/
│   ├── raw_generations_base.json       # 60 base model responses with token/timing metadata
│   ├── raw_generations_teacher.json    # 60 teacher responses with token/timing metadata
│   ├── raw_generations_student.json    # 60 student responses with token/timing metadata
│   ├── vram_measurements.json          # Peak VRAM per model recorded during nb04
│   ├── judge_scores.json               # 180 LLM judge scores (4 binary criteria × 60 queries × 3 models)
│   ├── eval_summary.json               # Aggregated metrics per model (feeds the README table)
│   ├── eval_metadata.json              # Evaluation config: judge model, date, generation params
│   └── teacher_generated_data.json     # 232 teacher-generated distillation examples from nb02
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
export HF_TOKEN=your_huggingface_token        # required for pushing models to the Hub
export ANTHROPIC_API_KEY=your_anthropic_key   # required for Notebook 05 (LLM judge)
```

On Windows (PowerShell): `$env:HF_TOKEN = "your_token"` / `$env:ANTHROPIC_API_KEY = "your_key"`

---

## Notebooks

Each notebook is self-contained and runs sequentially. Notebooks 1–4 require a GPU; Notebook 5 can run on CPU.

| Notebook | Description | GPU required |
|---|---|---|
| `01-teacher-finetuning` | Fine-tunes the teacher model with QLoRA | Yes |
| `02-data-generation` | Generates distillation dataset via teacher inference | Yes |
| `03-student-distillation` | Trains the student on teacher-generated data | Yes |
| `04-inference` | Runs all three models on 60 evaluation queries | Yes |
| `05-evaluation` | LLM judge scoring, analysis, and summary table | No |

All training notebooks were developed and run on **Kaggle** (T4 GPU, 16 GB VRAM).  
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/work/collections/18299508)

---

## Approach

### Fine-Tuning (QLoRA)
- The teacher (`LeoLM/leo-mistral-hessianai-7b-chat`) is fine-tuned with a QLoRA: the base model weights are quantized to 4-bit, and small trainable LoRA adapters (`r=16`) are attached to all attention and MLP projections.
- Only the adapter weights are updated — ~1% of total parameters — making training feasible on a single consumer GPU.
- The system prompt explicitly enforces the four-part response structure (opening formula → numbered steps → closing offer → formal register), mapping directly to the four criteria evaluated in Notebook 05.
- Training dataset: 28 synthetic German customer support Q&A pairs across categories including payments, shipping, returns, account security, and technical issues.

### Knowledge Distillation (Sequence-Level)
- This project uses **sequence-level / response-based distillation** (Kim & Rush, 2016), not classical logit-level KD (Hinton, 2015).
- Concretely: the fine-tuned teacher generates responses for 232 German support prompts. The student is then trained via standard SFT on these (prompt, teacher-response) pairs. The student never sees the teacher's internal probability distributions — only its generated outputs.
- This is a deliberate scope decision. Logit-level distillation (matching soft probability distributions via temperature-scaled KL divergence) would require both models in memory simultaneously and is left as a potential extension.

---

## Limitations

- **Dataset size:** The fine-tuning dataset is intentionally minimal (28 examples) — sufficient for style and format adaptation (response tone, language, structure), but not for teaching factual domain knowledge.
- **Synthetic data only:** All training data is synthetically generated. No real customer interactions were used.
- **Distillation method:** Sequence-level KD does not transfer the teacher's uncertainty or soft label information. Quality ceiling is the teacher's generation quality.
- **Hardware dependency:** Training requires a CUDA GPU; not reproducible on CPU.

---

## References

- [LeoLM](https://huggingface.co/LeoLM) — German-specialist LLMs by HessianAI
- Kim & Rush (2016) — [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947)
- Hinton et al. (2015) — [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Unsloth](https://github.com/unslothai/unsloth)

---

## Author

**Dennis Feyerabend** · May 2026  
[github.com/Feyerade](https://github.com/Feyerade) · [huggingface.co/Feyerade](https://huggingface.co/Feyerade)

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.  
Base model weights: `LeoLM/leo-mistral-hessianai-7b-chat` (Apache 2.0, HessianAI) and `Llama-3.2-1B-Instruct` (Llama 3.2 Community License, Meta).
