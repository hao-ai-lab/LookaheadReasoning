## Ablation Scripts

This folder contains scripts to run datasets with different methods for analyzing accuracy and acceptance rates.

### Prerequisites

Ensure vLLM servers (target, draft, judge) are running on the ports specified in `run_dataset.py`:

- `target_client`
- `draft_client`
- `judge_client`

#### Starting vLLM Servers

Start the three vLLM servers in separate terminals:

**Target Model**
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --port 12347 --enable-prefix-caching
```

**Draft Model**
```bash
CUDA_VISIBLE_DEVICES=1 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --port 12345 --enable-prefix-caching
```

**Judge Model**
```bash
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --enable-prefix-caching
```

### Run the dataset

Use `run_dataset.py` to process a question range with a selected method. The outputs (accuracy, predictions, logs, and judge results) will be written to a timestamped folder.

Examples:

```bash
python run_dataset.py --method baseline --start_qid 0 --end_qid 30

python run_dataset.py --method llm-j --start_qid 0 --end_qid 30

python run_dataset.py --method llm-j --allow_no_stop --start_qid 0 --end_qid 30

python run_dataset.py --method emb --start_qid 0 --end_qid 30

python run_dataset.py --method emb --allow_no_stop --start_qid 0 --end_qid 30
```

### Example accept rates

| Method | allow_no_stop | Overall accept rate |
|---|---|---|
| llm-j | Yes | 0.4573 |
| emb | Yes | 0.3664 |
| llm-j | No | 0.4322 |
| emb | No | 0.3688 |

Args:
- `--method`: algorithm variant to run (e.g., `baseline`, `llm-j`, `emb`).
- `--start_qid`, `--end_qid`: inclusive/exclusive range of question IDs to process.

### Analyze acceptance rate

Use `analyze_accept_rate.py` to compute acceptance rate statistics from judge outputs. Pass one or more run IDs (folder names or identifiers printed by `run_dataset.py`).

Example:

```bash
python analyze_accept_rate.py <path1> <path2>
```

The script will print summary metrics to stdout.


