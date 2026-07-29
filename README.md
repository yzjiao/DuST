<div align="center">

# DuST

### Primal Generation, Dual Judgment<br/>*Self-Training from Test-Time Scaling*

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat-square)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Model](https://img.shields.io/badge/base-Qwen3--30B--A3B-6633CC.svg?style=flat-square)](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507)
[![Benchmark](https://img.shields.io/badge/eval-LiveCodeBench%20v5%2Fv6-ff6b6b.svg?style=flat-square)](https://livecodebench.github.io/)

[**Key Idea**](#-key-idea) · [**Pipeline**](#-pipeline) · [**Quick Start**](#-quick-start) · [**Setup**](#-setup) · [**Usage**](#-detailed-usage) · [**Design**](#-design-choices)

<img src="assets/intro_figure.png" width="88%" alt="DuST pipeline overview"/>

</div>

> [!NOTE]
> Anonymous code release for double-blind review. Author, affiliation, and paper links are withheld and will be added in the camera-ready version.

---

## 💡 TL;DR

Test-time scaling reveals which candidates succeed and which fail — but this comparative signal is **discarded after inference**.

**DuST** (Dual Self-Training) recycles it: sample candidates, label via execution, and train the model to *rank* them by correctness using GRPO. The model is never directly rewarded for generating correct code, yet **both judgment and generation improve**.

<div align="center">

| Metric | Gain on LiveCodeBench |
|:--|:--|
| Best-of-4 accuracy | **+4.1** |
| pass@1 | **+3.1** |

</div>

A single rollout from the trained model matches the base model's Best-of-4.

---

## 🔑 Key Idea

Most test-time scaling methods rely on external verifiers or reward models. We show that the **same model** can serve as both generator and judge:

| | Role | What happens |
|:--:|:--|:--|
| **Primal** | Generation | The model generates `N=64` diverse solutions per coding problem, verified by a lightweight sandbox |
| **Dual** | Judgment | The model is trained via GRPO to rank candidate solutions by correctness — using only self-generated data as signal |

This creates a virtuous cycle:

```
better judgment → better selection at test time → better generation data → better judgment
```

---

## 🔁 Pipeline

| # | Stage | Command | Produces |
|:--:|:--|:--|:--|
| 0 | **Launch Sandbox** | `data_generation/0_launch_sandbox.py` | Docker containers for code execution |
| 1 | **Self-Distillation** | `data_generation/1_generate.py` | 64 solutions/problem + verified pass rates |
| 2 | **Process Data** | `data_generation/2_process_data.py` | Candidate sets of 4 |
| 3 | **Prepare Training** | `training/prepare_data.py` | Ranking prompts + labels |
| 4 | **GRPO Training** | `training/run_grpo.sh` | Trained judge (Megatron + verl) |
| 5 | **Evaluation** | `eval/sweep_checkpoints.py` | LiveCodeBench v5/v6 with selection |

---

## 🚀 Quick Start

```bash
# 1 — Install dependencies
pip install -r requirements.txt
cd third_party/verl       && pip install -e . && cd ../..
cd third_party/evalchemy  && pip install -e . && cd ../..

# 2 — Configure (edit this one file with your paths)
vim configs/paths.yaml

# 3 — Build sandbox & run the full pipeline
docker build -t code_sandbox:latest data_generation/sandbox_image/
python data_generation/0_launch_sandbox.py --num-instances 64
python data_generation/1_generate.py --shard-rank 0 --num-shards 1
python data_generation/2_process_data.py
python training/prepare_data.py
bash   training/run_grpo.sh
python eval/sweep_checkpoints.py --config eval/configs/sweep_lcbv6.yaml
```

---

## ⚙️ Setup

### Prerequisites

| Requirement | Notes |
|:--|:--|
| Python | 3.10+ |
| GPUs | CUDA 12.x, 8× A100/H100 recommended |
| Docker | Required for the code sandbox |
| AWS CLI | Optional — S3 checkpoint sync |

### Third-Party Dependencies

| Component | Purpose |
|:--|:--|
| [**verl**](https://github.com/volcengine/verl) | GRPO training with Megatron backend |
| [**evalchemy**](https://github.com/EvalAlchemy/evalchemy) | LiveCodeBench evaluation framework |

```bash
git clone https://github.com/volcengine/verl.git       third_party/verl
git clone https://github.com/EvalAlchemy/evalchemy.git third_party/evalchemy
```

### Configuration

All paths live in **one file** — [`configs/paths.yaml`](configs/paths.yaml):

```yaml
model:
  hf_id:        "Qwen/Qwen3-30B-A3B-Thinking-2507"
  local_cache:  "/mnt/models/Qwen3-30B-A3B-Thinking-2507"
dataset:
  raw_path:     "/path/to/coding_problems.parquet"
output:
  data_dir:       "./output/sd_data"
  checkpoint_dir: "./output/checkpoints"
sandbox:
  docker_image:  "code_sandbox:latest"
  num_instances: 64
```

Hyperparameters live in [`configs/data_gen.yaml`](configs/data_gen.yaml) (generation) and [`configs/training.yaml`](configs/training.yaml) (GRPO).

---

## 📖 Detailed Usage

<details open>
<summary><b>Stage 0 — Launch Sandbox</b></summary>

<br/>

A minimal sandbox server lives in `data_generation/sandbox_image/`:

```bash
docker build -t code_sandbox:latest data_generation/sandbox_image/
python data_generation/0_launch_sandbox.py --num-instances 64
```

This starts 64 containers (ports `8080–8143`), each exposing:

```http
POST /execute   {"code", "input", "timeout", "memory_limit_mb"}
             →  {"stdout", "stderr", "exit_code", "timed_out"}
```

</details>

<details>
<summary><b>Stage 1 — Self-Distillation Data Generation</b></summary>

<br/>

Generate 64 candidate solutions per problem with vLLM, verifying each against its test cases:

```bash
# Single GPU (debugging)
python data_generation/1_generate.py --shard-rank 0 --num-shards 1

# Multi-GPU / SLURM
python data_generation/1_generate.py --shard-rank $SLURM_ARRAY_TASK_ID --num-shards 64
```

</details>

<details>
<summary><b>Stages 2–3 — Data Processing</b></summary>

<br/>

```bash
# Group solutions into candidate sets of 4, filter trivial groups
python data_generation/2_process_data.py

# Convert to verl training format (ranking prompt + pass_rate labels)
python training/prepare_data.py
```

</details>

<details>
<summary><b>Stage 4 — GRPO Training</b></summary>

<br/>

```bash
# Single node (8 GPUs)
bash training/run_grpo.sh

# Multi-node
RAY_ADDRESS=auto NNODES=2 bash training/run_grpo.sh
```

</details>

<details>
<summary><b>Stage 5 — Evaluation</b></summary>

<br/>

```bash
# Single checkpoint
S3_CKPT_DIR=s3://.../global_step_100 bash eval/eval_checkpoint.sh

# Sweep all checkpoints with multiple repeats
python eval/sweep_checkpoints.py --config eval/configs/sweep_lcbv6.yaml
```

</details>

---

## 🧭 Design Choices

| Choice | Details |
|:--|:--|
| **Reward function** | Pairwise accuracy — fraction of correctly ordered candidate pairs |
| **RL algorithm** | GRPO with `n=8` rollouts per prompt |
| **Parallelism** | Megatron `EP=8` for MoE (128 experts, 3B active) |
| **Context length** | 32k prompt + 32k response |
| **Self-distillation** | Model generates its own training data (no teacher) |

### Model

- **Base** — [Qwen3-30B-A3B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507) · Mixture-of-Experts, 30B total / 3B active
- **Task** — given N code candidates for a problem, output a ranking from best to worst
- **Inference** — the trained judge selects the best among sampled solutions at test time

---

## 📁 File Structure

```
DuST/
├── configs/
│   ├── paths.yaml              # All paths (modify this)
│   ├── data_gen.yaml           # Generation hyperparameters
│   └── training.yaml           # GRPO training hyperparameters
│
├── data_generation/
│   ├── 0_launch_sandbox.py     # Start verification containers
│   ├── 1_generate.py           # Self-distillation pipeline
│   ├── 2_process_data.py       # Group candidates
│   ├── _dp_worker.py           # vLLM data-parallel worker
│   ├── sandbox_image/          # Dockerfile + server for sandbox
│   ├── utils/                  # Shared utilities
│   └── templates/              # Jinja2 prompt templates
│
├── training/
│   ├── run_grpo.sh             # GRPO launch script
│   ├── prepare_data.py         # Convert to verl format
│   └── reward_score.py         # Reward function (NDCG / PairAcc)
│
├── eval/
│   ├── eval_checkpoint.sh      # Single checkpoint eval
│   ├── sweep_checkpoints.py    # Multi-step sweep + aggregation
│   └── configs/                # LCB v5 / v6 sweep presets
│
└── third_party/
    ├── verl/                   # Training framework
    └── evalchemy/              # Eval framework
```

---

## 📄 Citation

> Withheld for double-blind review.

<div align="center">
<sub>Released under the Apache 2.0 License.</sub>
</div>
