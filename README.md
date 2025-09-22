
<div align="center">
    <h1>Scaling Speculative Decoding with Lookahead Reasoning</h1>
</div>

<div align="center" style="line-height: 1; margin-bottom: 12px;">
    | <a href="https://hao-ai-lab.github.io/blogs/lookaheadreasoning/">📝 Blog</a> 
    | <a href="https://arxiv.org/abs/2506.19830">📄 Paper</a> 
    | <a href="https://x.com/haoailab">🐦 Twitter/X</a> 
    |
</div>

This repository contains the official implementation of **Lookahead Reasoning**, a novel technique that accelerates large reasoning models by speculatively generating and verifying reasoning steps in parallel.

- [Method](#method)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Use Lookahead Reasoning](#how-to-use-lookahead-reasoning)
- [Citation](#citation)

----- 

## Method 

<div align="center">
    <img src="assets/LookaheadReasoningStep.jpg" alt="Lookahead Reasoning" width="600px">
</div>

**Lookahead Reasoning (LR)** is a novel method designed to accelerate the inference speed of large reasoning models (LRMs). While traditional token-level speculative decoding offers some speedup, it is bounded on complex, long-form tasks. This is because the probability of correctly guessing a long, exact sequence of tokens is exponentially low. Lookahead Reasoning mitigate this limitation by elevating speculation from the token level to the more abstract 'reasoning step' level.


The core of LR lies in a cyclical process where a smaller, faster 'draft' model generates multiple potential future steps, which are then verified in parallel by a powerful 'target' model. A single cycle unfolds as follows:

1.  **Drafting Future Steps:** A lightweight draft model proactively generates a sequence of multiple candidate future reasoning steps, such as ${\\hat{s}\_1, \\hat{s}\_2, \\hat{s}\_3}$.
2.  **Parallel Target Verification:** Instead of slow, sequential evaluation, the powerful target model processes all these proposals in a single, efficient **batched forward pass**. In this one pass, it generates its own "ground truth" version for each step ($s\_i$) conditioned on the *previous drafted step* ($\\hat{s}\_{i-1}$), allowing it to explore multiple potential reasoning paths simultaneously.
3.  **Semantic Verification:** A verifier compares each drafted step with its corresponding target-generated version (e.g., $\\hat{s}\_1$ vs. $s\_1$), checking for **semantic equivalence** rather than a perfect word-for-word match.
4.  **Acceptance and Correction:** The algorithm accepts the entire chain of semantically correct draft steps up to the first mismatch. At the point of divergence, it appends the target model's own generated step, ensuring correctness before the next cycle begins.

This process, which replaces multiple slow, sequential calls to the target model with a single parallel operation, is visualized below. In this example, the first two steps are accepted, and the third is corrected by the target model, producing three steps of reasoning for a fraction of the autoregressive cost.

By validating entire chunks of logic at once, LR can produce several correct reasoning steps for the latency cost of generating just one. Crucially, this step-level parallelism is **orthogonal** to existing token-level speculative decoding. This allows both methods to be combined, unlocking multiplicative speedups and a significant reduction in end-to-end generation time.


----- 

## Requirements

  * **Hardware:** A server with at least **4x 80GB GPUs** is required to run the example configuration (32B target/1.7B draft/7B judge).
  * **Software:** Python 3.10+, PyTorch, and `vLLM`.

-----

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/hao-ai-lab/LookaheadReasoning.git
cd LookaheadReasoning
uv pip install -e .
```

-----

## How to Use Lookahead Reasoning

Our framework operates with three key components:

1.  **Target Model:** The large, powerful model you want to accelerate (e.g., Qwen3-32B).
2.  **Draft Model:** A smaller, faster model that generates speculative steps (e.g., Qwen3-1.7B).
3.  **Judge Model:** A verifier model that assesses if the draft steps are semantically same as the target steps using LLM-as-a-judge (e.g., Qwen2.5-7B-Instruct).

The following steps guide you through running an experiment using the Qwen model family.

### Step 1: Launch the Judge Model

First, launch the Judge model as a dedicated inference server using `vLLM`. This service will be called by the main script. Run this command in a separate terminal session.

> **Note:** The `--enable-prefix-caching` flag is recommended for better performance.

```bash
# Assigns the Judge model to GPU 3
CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen2.5-7B-Instruct --enable-prefix-caching
```

### Step 2: Run the Main Experiment Script

Next, execute the main script to start the generation process. The script will automatically handle the orchestration between the Target and Draft models.

The example commands below distribute the models as follows:

  * **Target Model (32B):** Hosted on `GPU0` and `GPU1`.
  * **Draft Model (1.7B):** Hosted on `GPU3`.

You can run experiments in four different modes:

#### **Mode 1: Lookahead Reasoning (LR) - *Our Method***

This enables our core Lookahead Reasoning algorithm.

```bash
python main.py --dataset data/aime-2024.jsonl --use_spec
```

#### **Mode 2: LR with Speculative Decoding (LR+SD)**

This combines Lookahead Reasoning with n-gram based speculative decoding for potentially even greater speedups.

```bash
python main.py --dataset data/aime-2024.jsonl --use_spec --enable_n_gram
```

#### **Mode 3: Standard Speculative Decoding (SD)**

Runs inference using only traditional n-gram based speculative decoding.

```bash
python main.py --dataset data/aime-2024.jsonl --enable_n_gram
```

#### **Mode 4: Baseline**

Runs standard autoregressive decoding with the target model, without any acceleration.

```bash
python main.py --dataset data/aime-2024.jsonl
```

### Step 3: Collect Results

The results, including generated outputs and performance metrics, will be saved to a results file (e.g., a `.jsonl` file) in the project directory. Check the script's output logs for the exact filename and location.


-----


### Reproduced Results on AIME24

**Setup:** 480 total traces (16 samples per problem) on 4×H200  
**Models:** Qwen3-32B (target) / Qwen3-1.7B (draft) / Qwen2.5-7B-Instruct (judge)

| Method | Accuracy | Speed | Speedup |
|--------|----------|-------|---------|
| Autoregressive | 80.21% | 76.38 | 1.00x |
| Ours | 81.04% | 90.90 | **1.19x** |
| Ours + SD | 81.25% | 120.74 | **1.58x** |

Our method achieves significant acceleration while preserving accuracy, with multiplicative gains when combined with speculative decoding.

**Note:** Use `report_prefix.py` to aggregate results across folders with the same prefix.

-----

### Configuration & Arguments

This section details the command-line arguments available in `main.py` for customizing your experiments.

#### **Data & Sampling Arguments**

  * `--dataset`: Path to the dataset file. **Default:** `aime-2024.jsonl`.
  * `--start_qid`: The starting question ID (index) from the dataset to process.
  * `--end_qid`: The ending question ID (index) from the dataset to process.
  * `--prefix`: A prefix string used for naming output files. **Default:** `AIME`.

#### **Model & GPU Arguments**

  * `--model`: Hugging Face path for the **Target Model**. **Default:** `Qwen/Qwen3-32B`.
  * `--draft_model`: Hugging Face path for the **Draft Model**. **Default:** `Qwen/Qwen3-1.7B`.
  * `--judge_model`: Hugging Face path for the **Judge Model**. **Default:** `Qwen/Qwen2.5-7B-Instruct`.
  * `--judge_port`: Network port for the vLLM Judge model server. **Default:** `8000`.
  * `--target_gpu_id`: Comma-separated list of GPU IDs for the Target Model. **Default:** `0,1`.
  * `--draft_gpu_id`: GPU ID for the Draft Model. **Default:** `2`.

#### **Lookahead Reasoning (LR) Arguments**

  * `--use_spec`: **The main flag to enable Lookahead Reasoning.**
  * `--max_depth`: The maximum depth of the reasoning tree to explore. **Default:** `4`.
  * `--width`: The number of candidate sequences (beams) to generate at each step of the tree. **Default:** `1`.
  * `--ignore_half_sentence`: If set, prevents the generation from stopping in the middle of a sentence, improving coherence.

#### **Speculative Decoding (SD) Arguments**

  * `--enable_n_gram`: **The flag to enable n-gram based Speculative Decoding.** Can be used alone or with LR.
  * `--num_speculative_tokens`: Number of tokens to generate speculatively in n-gram SD. **Default:** `6`.
  * `--prompt_lookup_max`: Maximum n-gram length to match against the prompt history. **Default:** `2`.

#### **General Generation Arguments**

  * `--max_tokens_len`: Maximum number of tokens for the entire sequence (prompt + completion). **Default:** `37000`.

-----


## Citation

If you find Lookahead Reasoning useful in your research or project, please cite our paper:

```bibtex
@article{fu2025scaling,
  title={Scaling Speculative Decoding with Lookahead Reasoning},
  author={Fu, Yichao and Ge, Rui and Shao, Zelei and Deng, Zhijie and Zhang, Hao},
  journal={arXiv preprint arXiv:2506.19830},
  year={2025}
}
```
