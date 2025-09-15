import openai
import asyncio
from tqdm import tqdm
import time
from transformers import AutoTokenizer
import os
import json
import datetime
import argparse
import random
import subprocess
import requests

from lookahead_reasoning import Targeter, Drafter
from lookahead_reasoning import TreeNode

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='aime-2024.jsonl')
parser.add_argument('--start_qid', type=int, default=None)
parser.add_argument('--end_qid', type=int, default=None)
parser.add_argument('--prefix', type=str, default='AIME')

parser.add_argument('--max_depth', type=int, default=4)
parser.add_argument('--width', type=int, default=1)

parser.add_argument('--model', type=str, default='Qwen/Qwen3-32B')
parser.add_argument('--draft_model', type=str, default='Qwen/Qwen3-1.7B')
parser.add_argument('--judge_model', type=str, default='Qwen/Qwen2.5-7B-Instruct')
parser.add_argument('--judge_port', type=int, default=8001)

parser.add_argument('--target_gpu_id', type=str, default='0,1')
parser.add_argument('--draft_gpu_id', type=str, default='2')

parser.add_argument('--enable_n_gram', action='store_true')
parser.add_argument('--num_speculative_tokens', type=int, default=6)
parser.add_argument('--prompt_lookup_max', type=int, default=2)

parser.add_argument('--ignore_half_sentence', action='store_true')
parser.add_argument('--max_tokens_len', type=int, default=37000)
parser.add_argument('--use_spec', action='store_true')

args = parser.parse_args()

judge_client = openai.AsyncOpenAI(base_url=f"http://127.0.0.1:{args.judge_port}/v1", api_key="None", timeout=None)


MODEL_CONFIGS = {
    'deepseek': {
        'name': 'deepseek',
        'temperature': 0.6,
        'top_p': 0.95,
        'top_k': 0,
        'max_tokens': 32768,
        'prompt_template': 'deepseek',
        'eos': "<｜end▁of▁sentence｜>",
        'eos_id': 151643,
        'stop': ['\n\n'],
        'step_tokens': 100,
    },
    'qwen3': {
        'name': 'qwen3',
        'temperature': 0.6,
        'top_p': 0.95,
        'top_k': 20,
        'max_tokens': 38912,
        'prompt_template': 'qwen3',
        'eos': '<|endoftext|>',
        'eos_id': [151643, 151645],
        'stop': ['\n\n'],
        'step_tokens': 100,
    },
    'gpt': {
        'name': 'gpt',
        'temperature': 1.0,
        'top_p': 1.0,
        'top_k': 40,
        'max_tokens': 130000,
        'prompt_template': 'gpt',
        'eos': '<|endoftext|>',
        'stop': ['\n\n'],
        'step_tokens': 100,

    }
}


def get_model_config(model_name):
    """Get configuration for a specific model."""
    model_name_lower = model_name.lower()
    
    # Match model configurations
    if 'deepseek-r1' in model_name_lower:
        return MODEL_CONFIGS['deepseek']
    elif 'qwen3' in model_name_lower:
        return MODEL_CONFIGS['qwen3']
    elif 'gpt' in model_name_lower or 'oai' in model_name_lower:
        return MODEL_CONFIGS['gpt']
    else:
        assert False, f"Unknown model: {model_name}"



async def main():

    output_dir = args.prefix + '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '_' + str(random.randint(100000, 999999))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    questions = load_questions(args.dataset)[args.start_qid:args.end_qid]

    target_tokenizer = AutoTokenizer.from_pretrained(args.model)
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_model)

    target_config = get_model_config(args.model)
    draft_config = get_model_config(args.draft_model)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.target_gpu_id
    # Set environment variables for GPU usage
    target_model = Targeter(args.model, eos=None, eos_id=target_config['eos_id'], target_gpu_id=args.target_gpu_id,
                    enable_n_gram=args.enable_n_gram, vllm_config={'force_eager': False, 'num_speculative_tokens': args.num_speculative_tokens, 'prompt_lookup_max': args.prompt_lookup_max}) 

    os.environ['CUDA_VISIBLE_DEVICES'] = args.draft_gpu_id
    draft_model = Drafter(args.draft_model, eos=None, eos_id=draft_config['eos_id'], draft_gpu_id=args.draft_gpu_id,
                    enable_n_gram=args.enable_n_gram, vllm_config={'force_eager': False, 'num_speculative_tokens': args.num_speculative_tokens, 'prompt_lookup_max': args.prompt_lookup_max})


    assert target_config['name'] == draft_config['name'], \
        "Target and draft models must be of the same type (e.g., both Qwen3 or both GPT)."

    target_config['judge_model'] = args.judge_model
    print(f"Target Model Config: {target_config}")
    print(f"Draft Model Config: {draft_config}")

    for i in range(len(questions)):
        await run_problem(questions[i], i, target_model, draft_model, \
                          target_tokenizer, draft_tokenizer, \
                          target_config, draft_config, output_dir)

    print(f"Results saved to {output_dir}")


if __name__ == '__main__':

    asyncio.run(main())

