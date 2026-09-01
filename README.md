# Distil Support LLM — German Customer Support via Knowledge Distillation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![HuggingFace – Teacher](https://img.shields.io/badge/HuggingFace-Teacher%20Adapter-orange)](https://huggingface.co/Feyerade/german-support-leollm-lora-adapter)
[![HuggingFace – Student](https://img.shields.io/badge/HuggingFace-Student%20Model-orange)](https://huggingface.co/Feyerade/german-support-llama-1b-distilled)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Open in HF Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/Feyerade/german-support-llm-demo)

> **Showcase project.** Built as a portfolio piece demonstrating QLoRA fine-tuning and knowledge distillation on a domain-specific task. The pipeline is end-to-end reproducible via the notebooks and pre-trained models on Hugging Face.

**[▶ Try the live demo](https://huggingface.co/spaces/Feyerade/german-support-llm-demo)**

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

- Evaluation runs all three models on 120 German customer-support queries and scores each response with Claude Haiku as a blind LLM judge (4 binary criteria).
- Scored criteria: an opening acknowledgement, a clearly numbered list of steps, a closing sentence, and overall professional tone.
- Three queries are excluded because the teacher response degenerated into a runaway enumeration, leaving 117 queries in the analysis.
- Judging is in `notebooks/05-evaluation.ipynb`, all analysis in `notebooks/06-statistical-analysis.ipynb`.

**Table 1: Overview over general score distributions**

| Metric                 | Base (1B) | Teacher (7B + LoRA) | Student (1B distilled) |
|------------------------|---|---|---|
| Avg Format Score (0–4) | 2.09 | 3.50 | 3.27 |
| Acknowledgement        | 70.1% | 75.2% | 89.7% |
| Structured Steps       | 54.7% | 85.5% | 71.8% |
| Closing Sentence       | 31.6% | 95.7% | 94.0% |
| Professional Tone      | 53.0% | 94.0% | 71.8% |
| Avg Word Count         | 79.0 | 63.7 | 58.0 |
| Avg Tokens/sec (T4)    | 45.3 | 15.8 | 45.3 |
| Peak VRAM (4-bit)      | 1.15 GB | 4.68 GB | 1.15 GB |

### Agreement with the teacher

Compliance rates alone are hard to read: not every query warrants a numbered list, so the ideal rate is unknown.   
The analysis therefore takes the teacher as the reference and measures, query by query, whether a model makes the same call. This method has a known maximum of 100% (completely agree with teacher) and is comparable across criteria.  Paired over 117 queries.

**Table 2: How close is the student to the teacher?**

| Criterion | Student agrees | 95% CI | Student only applied rule | Teacher only applied rule |
|---|---|---|---|---|
| Acknowledgement  | 73.5% | [64.9%, 80.7%] | 24 | 7 |
| Structured Steps | 69.2% | [60.4%, 76.9%] | 10 | 26 |
| Closing Sentence | 91.5% | [85.0%, 95.3%] | 4 | 6 |
| Professional Tone| 69.2% | [60.4%, 76.9%] | 5 | 31 |

The last two columns count the queries where the two disagree, split by direction.

**Table 3: Is the student closer to the teacher than the untrained base model?**

| Criterion | Base agrees | Student agrees | Difference | 95% CI | Informative | p (Holm) |
|---|---|---|---|---|---|---|
| Acknowledgement  | 59.0% | 73.5% | +14.5 pp | [+6.0, +23.1] | 31 | 0.010 |
| Structured Steps | 59.0% | 69.2% | +10.3 pp | [−1.7, +22.2] | 52 | 0.126 |
| Closing Sentence | 30.8% | 91.5% | +60.7 pp | [+50.4, +70.1] | 77 | <0.001 |
| Professional Tone| 55.6% | 69.2% | +13.7 pp | [+3.4, +23.9] | 38 | 0.028 |

"Informative" counts the queries where exactly one of the two models matched the teacher. Only those carry information about a difference, and the test uses only those. Intervals are bootstrap intervals over queries, tests are exact McNemar with Holm correction across the four criteria.

### **Key findings**

**Format transfer works at a fraction of the cost**
- The student reaches 3.27 of the teacher's 3.50 format score, at 45.3 tokens/sec on 1.15 GB against 15.8 tokens/sec on 4.68 GB.

**Closing sentences transfer almost completely**
- 91.5% agreement with the teacher, 60.7 percentage points above the base model.
- The largest and most reliable effect in the evaluation, resting on 77 informative queries.

**Professional tone and structured steps transfer only partially**
- Both sit at 69.2% agreement with the teacher (table 2).
- Both deviate in one direction: the student omits the behaviour where the teacher applies it, 31 times for tone and 26 times for structured steps, rather than over-applying it (table 2).
- These two failures are what a further training round would need to target.

**Acknowledgement goes the other way**
- The student opens empathetically more often than the teacher (table 1: 89.7% against 75.2%).
- With the same system prompt, the base model already produces an opener in 70.1% of responses, close to the teacher's own 75.2% (table 1). The student keeps that habit and adds the teacher's structure on top.

**Structured steps are not statistically established**
- The improvement over the base model is +10.3 percentage points, but the interval runs from −1.7 to +22.2 (table 3).
- The data cannot separate a real gain from none. Reported as inconclusive rather than smoothed over.

---

## Repo Structure

```
distil-support-llm/
├── notebooks/
│   ├── 01-teacher-finetuning.ipynb      # QLoRA fine-tuning of the teacher model
│   ├── 02-data-generation.ipynb         # Teacher generates the distillation dataset
│   ├── 03-student-distillation.ipynb    # Student training on teacher outputs
│   ├── 04-inference.ipynb               # Runs all three models on 120 evaluation queries
│   ├── 05-evaluation.ipynb              # LLM judge scoring only
│   └── 06-statistical-analysis.ipynb    # All analysis: overview, agreement, tests
├── results/
│   ├── raw_generations_base.json        # 120 base model responses with token/timing metadata
│   ├── raw_generations_teacher.json     # 120 teacher responses with token/timing metadata
│   ├── raw_generations_student.json     # 120 student responses with token/timing metadata
│   ├── vram_measurements.json           # Peak VRAM per model recorded during nb04
│   ├── judge_scores.json                # 360 LLM judge scores (4 binary criteria × 120 queries × 3 models)
│   ├── eval_summary.json                # Aggregated metrics per model (feeds the README table)
│   ├── agreement_summary.json           # Agreement analysis with intervals, tests and settings
│   ├── eval_metadata.json               # Evaluation config: judge model, date, generation params
│   └── teacher_generated_data.json      # 232 teacher-generated distillation examples from nb02
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

Each notebook is self-contained and runs sequentially. Notebooks 1 to 4 require a GPU, Notebooks 5 and 6 run on CPU.

| Notebook | Description | GPU required |
|---|---|---|
| `01-teacher-finetuning` | Fine-tunes the teacher model with QLoRA | Yes |
| `02-data-generation` | Generates distillation dataset via teacher inference | Yes |
| `03-student-distillation` | Trains the student on teacher-generated data | Yes |
| `04-inference` | Runs all three models on 120 evaluation queries | Yes |
| `05-evaluation` | LLM judge scoring, writes `judge_scores.json` | No |
| `06-statistical-analysis` | Overview table, agreement analysis, tests | No |

Scoring and analysis are separate so that adding an analysis never requires paying for a new judge run.

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
- **Excluded queries:** Three of 120 queries were dropped because the teacher response collapsed into a repeating enumeration, which makes it useless as a reference. The exclusion was decided from the teacher responses alone, before any comparison was run.
- **Agreement is not quality:** The analysis measures whether the student makes the same formatting decision as the teacher, not whether that decision was correct. A student copying a weak teacher habit counts as success here. The judge scores form, not factual accuracy.
- **What the agreement rates support:** The student's agreement sits close to what its overall application rates alone would produce. The evidence supports "the student adopted the teacher's formatting behaviour", not "the student reproduces the teacher's decisions query by query".
- **Single judge:** One scoring pass by one model at temperature 0. Reproducible, but not validated against a second rater.

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
