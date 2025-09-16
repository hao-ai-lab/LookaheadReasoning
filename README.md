# LookaheadReasoning

## Installation

Install vLLM
```bash
git clone git@github.com:hao-ai-lab/LookaheadReasoning.git
cd LookaheadReasoning
pip install -e .
```

In another terminal:
```bash
CUDA_VISIBLE_DEVICES=5 vllm serve Qwen/Qwen2.5-7B-Instruct --enable-prefix-caching
```

## Usage

**LR:**
```bash
python main.py --dataset data/aime-2024.jsonl --use_spec --ignore_half_sentence
```

**LR+SD:**
```bash
python main.py --dataset data/aime-2024.jsonl --use_spec --ignore_half_sentence --enable_n_gram
```

**SD:**
```bash
python main.py --dataset data/aime-2024.jsonl --enable_n_gram
```

**Baseline:**
```bash
python main.py --dataset data/aime-2024.jsonl
```

**To specify samples in the dataset:**
```bash
python main.py --dataset data/aime-2024.jsonl --start_qid 0 --end_qid 1 --use_spec --ignore_half_sentence
```