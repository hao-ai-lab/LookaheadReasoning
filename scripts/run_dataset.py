import json
import csv
import time
import uuid
from openai import OpenAI
from tqdm import tqdm
import re
import argparse
from dynasor.core.evaluator import (
    extract_answer,
    strip_string,
    math_equal,
    extract_first_boxed_answer,
)
import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
import threading
from sentence_transformers import SentenceTransformer

def load_questions(file_path):
    """Load questions from jsonl file"""
    questions = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            questions.append(data)
    return questions

def get_model_response(prompt, model="gpt-4-turbo-preview", temperature=0.0, max_tokens=1000, stop=None, client=None, method="baseline"):
    """Get response from OpenAI API"""
    try:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            top_p=0.95
        )
        finish_reason = response.choices[0].finish_reason
        if method == 'llm-j' or method == 'emb':
            if hasattr(response.choices[0], 'stop_reason'):
                stop_reason = response.choices[0].stop_reason
            else:
                stop_reason = response.choices[0].matched_stop
        elif method == 'baseline':
            stop_reason = response.choices[0].stop_reason
        # Get the response content
        completion_text = response.choices[0].text
        return completion_text, finish_reason, stop_reason
    except Exception as e:
        print(f"Error getting response: {e} {model}")
        return None



def get_model_response_chat(prompt, model="gpt-4-turbo-preview", temperature=0.0, max_tokens=1000, stop=None, client=None):
    """Get response from OpenAI API"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            top_p=0.95
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting response: {e} {model}")
        return None


def save_results(questions, responses, output_file):
    """Save results to CSV file"""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Question', 'Response'])
        for q, r in zip(questions, responses):
            writer.writerow([q['question'], r])

equal_prompts = [
'''<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nEvaluate whether the following two reasoning steps (s1 and s2) convey exactly the same meaning. Focus on semantic similarity rather than exact wording. 

Compare the main ideas, key points, overall message, logical structure, and numerical calculations/results of both reasoning steps.

If the reasoning steps convey essentially the same meaning and generate same calculation results, respond with [aligned].
If the reasoning steps express different meanings, respond with [unaligned]. If it is too hard to determine, respond with [unaligned]

Please directly provide the final result in [aligned] or [unaligned].

Reasoning step 1 (s1):
<start_s1>
{}
<end_s1>

Reasoning step 2 (s2):
<start_s2>
{}
<end_s2><|im_end|>\n<|im_start|>assistant\n['''
]


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='llm-j', 
                    choices=['llm-j', 'emb', 'baseline'],
                    help='Method to run: llm-j (LLM Judge), emb (Embedding), baseline')
    parser.add_argument('--model', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', help='Model path')
    parser.add_argument('--judge_model', type=str, default='Qwen/Qwen2.5-7B-Instruct', help='Judge model path')
    parser.add_argument('--dataset', type=str, default='../data/aime-2024.jsonl', help='Dataset path')
    parser.add_argument('--prefix', type=str, default='judge', help='Prefix')
    parser.add_argument('--start_qid', type=int, default=None, help='Start question id')
    parser.add_argument('--end_qid', type=int, default=None, help='End question id')
    parser.add_argument('--prompt_idx', type=int, default=0, help='Prompt index')
    parser.add_argument('--threshold', type=float, default=0.95, help='Prompt index')
    parser.add_argument('--allow_no_stop', action='store_true', help='Allow no stop')
    parser.add_argument('--max_workers', type=int, default=20, help='Max workers')
    parser.add_argument('--max_samples', type=int, default=1, help='Max samples')
    return parser.parse_args()


def initialize_clients(args):
    """Initialize OpenAI clients for target, draft, and judge models"""
    target_client = [OpenAI(base_url=f"http://127.0.0.1:12347/v1", api_key="None", timeout=100000)]
    
    draft_client = None
    judge_client = None
    if args.method == 'llm-j' or args.method == 'emb':
        draft_client = [OpenAI(base_url=f"http://127.0.0.1:12345/v1", api_key="None", timeout=100000)]
        judge_client = [OpenAI(base_url=f"http://127.0.0.1:8000/v1", api_key="None", timeout=100000)]
    
    return target_client, draft_client, judge_client


def initialize_tokenizer(args):
    """Initialize tokenizer if needed"""
    tokenizer = None
    if args.method == 'llm-j' or args.method == 'emb':
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    return tokenizer


def setup_output_directory(prefix):
    """Create output directory with timestamp"""
    run_prefix = prefix + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = run_prefix
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def setup_embedding_model():
    """Setup embedding model and similarity function for 'emb' method"""
    embedding_device = "cuda:2"
    print(f"Loading embedding model to {embedding_device}")
    
    try:
        embedding_model = SentenceTransformer("all-mpnet-base-v2", device=embedding_device)
        print(f"Successfully loaded embedding model to {embedding_device}")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        embedding_model = None
    
    embedding_lock = threading.Lock()
    
    def compute_similarity(sentence1, sentence2):
        """Compute cosine similarity between two sentences"""
        if embedding_model is None:
            print("Warning: Embedding model not loaded, returning 0 similarity")
            return 0.0
            
        with embedding_lock:
            try:
                embeddings = embedding_model.encode([sentence1, sentence2])
                similarity = embedding_model.similarity(embeddings[0], embeddings[1])
                return similarity
            except Exception as e:
                print(f"Error computing similarity: {e}")
                return 0.0
    
    return compute_similarity


def process_questions_parallel(questions, args, target_client, draft_client, judge_client, 
                                tokenizer, output_dir, compute_similarity=None, equal_prompt=None):
    """Process questions in parallel using ThreadPoolExecutor"""
    temperature = 0.6
    results = []
    
    def process_question(q, sample_idx, question_idx):
        if 'question' not in q:
            q['question'] = q['problem']
        if 'id' not in q:
            q['id'] = question_idx

        inp = '<｜User｜>' + q['question'] + '<｜Assistant｜>'
        next_sentence = ""
        generations = []
        generation_target = []
        generation_draft = []
        accepts = []
        equals = []
        if args.method == 'llm-j' or args.method == 'emb':
            infos = []

            token_length = 0
            while True: 
                inp = inp + next_sentence
                if args.method == 'llm-j':
                    sentence_target, finish_reason, stop_reason = get_model_response(inp, temperature=temperature, max_tokens=100, stop=['\n\n'], client=target_client[0], model=args.model, method=args.method)
                elif args.method == 'emb':
                    sentence_target, finish_reason, stop_reason = get_model_response(inp, temperature=temperature, max_tokens=100, stop=['\n\n'], client=target_client[0], model=args.model, method=args.method)
                generation_target.append(sentence_target)

                if finish_reason == 'stop' and stop_reason != '\n\n':
                    next_sentence = sentence_target
                    generations.append(next_sentence)
                    break
                sentence_draft, finish_reason_draft, stop_reason_draft = get_model_response(inp, temperature=temperature, max_tokens=100, stop=['\n\n'], client=draft_client[0], model='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', method=args.method)
                generation_draft.append(sentence_draft)

                if args.method == 'llm-j':
                    if args.prompt_idx == -1:
                        equal, _, _ = get_model_response(equal_prompt.format(sentence_target.strip(), sentence_draft.strip()), temperature=0.0, max_tokens=1000, client=judge_client[0], model=args.judge_model)

                        print('xEqual: ', equal)
                        if '[aligned]' in equal and '[unaligned]' not in equal and finish_reason == 'stop' and finish_reason_draft == 'stop':
                            next_sentence = sentence_draft 
                            if finish_reason_draft == 'stop' and stop_reason_draft == '\n\n':
                                next_sentence += '\n\n'
                            accepts.append(1)
                        else:
                            next_sentence = sentence_target
                            if finish_reason == 'stop' and stop_reason == '\n\n':
                                next_sentence += '\n\n'
                            accepts.append(0)

                    else:
                        equal, _, _ = get_model_response(equal_prompt.format(sentence_target.strip(), sentence_draft.strip()), temperature=0.0, max_tokens=1, client=judge_client[0], model=args.judge_model)
                        print('Equal: ', equal)
                        if 'ali' in equal and 'un' not in equal and (args.allow_no_stop or (finish_reason == 'stop' and stop_reason == '\n\n')):
                            next_sentence = sentence_draft 
                            if finish_reason_draft == 'stop' and stop_reason_draft == '\n\n':
                                next_sentence += '\n\n'
                            accepts.append(1)
                        else:
                            next_sentence = sentence_target
                            if finish_reason == 'stop' and stop_reason == '\n\n':
                                next_sentence += '\n\n'
                            accepts.append(0)

                elif args.method == 'emb':
                    equal = compute_similarity(sentence_target.strip(), sentence_draft.strip()).item()
                    print('Equal: ', equal)
                    if equal > args.threshold and (args.allow_no_stop or (finish_reason == 'stop' and stop_reason == '\n\n')):
                        next_sentence = sentence_draft 
                        if finish_reason_draft == 'stop' and stop_reason_draft == '\n\n':
                            next_sentence += '\n\n'
                        accepts.append(1)
                    else:
                        next_sentence = sentence_target
                        if finish_reason == 'stop' and stop_reason == '\n\n':
                            next_sentence += '\n\n'
                        accepts.append(0)

                generations.append(next_sentence)
                infos.append((equal, generation_target[-1], generation_draft[-1]))
                equals.append(equal)

                tokens = tokenizer.encode(next_sentence, add_special_tokens=False)
                token_length += len(tokens)
            
                if token_length > 32768:
                    print(f"Token length ({token_length}) exceeds 16000, truncating input")
                    break

        elif args.method == 'baseline':
            next_sentence, finish_reason, stop_reason = get_model_response(inp, temperature=temperature, max_tokens=32000, client=target_client[0], model=args.model)
        
        inp = inp + next_sentence
        print('Done ', question_idx)
        
        # Save final input and answer to JSON file
        if args.method == 'llm-j' or args.method == 'emb':
            output_data = {
                'question': q['question'],
                'final_input': inp,
                'answer': inp,
                'accepts': accepts,
                'equals': equals,
                'generations_target': generation_target,
                'generations_draft': generation_draft,
                'generations': generations,
                'gold': q['answer'],
                'infos': infos,
            }
        elif args.method == 'baseline':
            output_data = {
                'question': q['question'],
                'final_input': inp,
                'answer': inp,
                'accepts': accepts,
                'equals': equals,
                'generations_target': generation_target,
                'generations_draft': generation_draft,
                'generations': generations,
                'gold': q['answer'],
            }
        with open(output_dir + '/' + str(q["id"]) + '_' + str(sample_idx) + '.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        print('Question: ', q['id'], 'answer:', extract_answer(inp, 'aime'), 'gold:', q['answer'], 'Spec: ', inp is None)
        return {'answer': extract_answer(inp, 'aime'), 'gold': q['answer']}

    # Process questions in parallel
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for sample_idx in range(args.max_samples):
            print('Running question', sample_idx, len(questions))
            future_to_question = {executor.submit(process_question, q, sample_idx, question_idx): q 
                                 for question_idx, q in enumerate(questions)}
            
        for future in tqdm(as_completed(future_to_question), total=len(questions) * args.max_samples):
            question = future_to_question[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Question processing failed: {e}")
    
    return results


def calculate_and_save_accuracy(results, questions, output_dir):
    """Calculate accuracy and save results"""
    correct = 0
    for result in results:
        print(result['answer'], result['gold'])
        if math_equal(str(result['answer']), str(result['gold'])):
            correct += 1
    
    accuracy = correct / len(results) if results else 0
    print(f"Accuracy: {accuracy}")
    
    accuracy_data = {
        'accuracy': correct / len(questions),
        'results': len(results)
    }
    with open(output_dir + '/' + 'accuracy.json', 'w') as f:
        json.dump(accuracy_data, f, indent=2)

    # Save results to CSV file
    with open(output_dir + '/' + 'results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['answer', 'gold'])
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def main():
    """Main function to run dataset processing"""
    args = parse_arguments()
    
    # Initialize clients
    target_client, draft_client, judge_client = initialize_clients(args)
    
    # Initialize tokenizer
    tokenizer = initialize_tokenizer(args)
    
    # Setup output directory
    output_dir = setup_output_directory(args.prefix)
    
    # Setup embedding model if needed
    compute_similarity = None
    if args.method == 'emb':
        compute_similarity = setup_embedding_model()
    
    # Load questions
    questions = load_questions(args.dataset)[args.start_qid:args.end_qid]
    
    # Get equal prompt if needed
    equal_prompt = None
    if args.method == 'llm-j':
        equal_prompt = equal_prompts[args.prompt_idx]
    
    # Process questions in parallel
    results = process_questions_parallel(
        questions, args, target_client, draft_client, judge_client,
        tokenizer, output_dir, compute_similarity, equal_prompt
    )
    
    # Calculate and save accuracy
    calculate_and_save_accuracy(results, questions, output_dir)

if __name__ == "__main__":
    main()
