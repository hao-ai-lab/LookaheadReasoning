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

# Initialize OpenAI client


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
<end_s2><|im_end|>\n<|im_start|>assistant\n[''',    
    
'''<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nAnalyze the two paragraphs below and determine if they convey the same meaning. Focus on semantic similarity rather than exact wording. 

Compare the main ideas, key points, overall message, and calculation results of both paragraphs.

If the paragraphs convey essentially the same meaning and generate same calculation results, respond with [aligned].
If the paragraphs express different meanings, respond with [unaligned].
If it is too hard to determine, respond with [unaligned].

Please directly provide the final result in [aligned] or [unaligned].
<paragraph1>
{}
</paragraph1>
<paragraph2>
{}
</paragraph2>
<|im_end|>\n<|im_start|>assistant\n[''' ,

'''<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nAnalyze the two paragraphs below and determine if they convey exactly the same meaning. Apply strict criteria for semantic equivalence and identical calculation results.

Compare the following aspects with precision:
1. Main ideas and central claims
2. Key supporting points and evidence
3. Overall message and implications
4. Mathematical calculations and numerical results (if present)
5. Logical structure and reasoning

If the paragraphs convey identical meaning with matching calculation results, respond with [aligned].
If there are ANY differences in meaning, conclusions, or numerical results, respond with [unaligned].

Do not provide explanations or analysis - output ONLY [aligned] or [unaligned].

<paragraph1>
{}
</paragraph1>
<paragraph2>
{}
</paragraph2><|im_end|>\n<|im_start|>assistant\n[''',

'''<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nEvaluate whether the following two reasoning steps (s1 and s2) convey exactly the same meaning.

Determine if s1 and s2 are semantically equivalent via mutual entailment:
- [aligned] if s1 entails s2 AND s2 entails s1
- [unaligned] if any semantic difference prevents full mutual derivation

Key criteria:
1. Evaluate semantic equivalence only, not factual correctness
2. Assess all logical implications/inferences
3. Disregard superficial wording differences
4. Verify identical core reasoning, concept relationships, and logical structure
5. Don't evaluate truth value in reality
6. Check mutual entailment exclusively
7. [unaligned] if any expressions/symbols/notations appear in only one statement
8. [unaligned] if one statement contains additional information or explanations not present in the other
9. Ensure exact matching of all numerical calculations/results
10. If it is too hard to determine, respond with [unaligned]

Reasoning step 1 (s1):
<start_s1>
{}
<end_s1>

Reasoning step 2 (s2):
<start_s2>
{}
<end_s2>

Do not provide explanations or analysis - output ONLY [aligned] or [unaligned].<|im_end|>\n<|im_start|>assistant\n[''',

'''<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nAnalyze the two paragraphs below and determine if they convey the same meaning. Focus on semantic similarity rather than exact wording. 

Compare the main ideas, key points, overall message, and calculation results of both paragraphs.

If the paragraphs convey essentially the same meaning and generate same calculation results, respond with [aligned].
If the paragraphs express different meanings, respond with [unaligned].

Please reason it step by step and provide the final result in [aligned] or [unaligned].
<paragraph1>
{}
</paragraph1>
<paragraph2>
{}
</paragraph2>
<|im_end|>\n<|im_start|>assistant\n[''' ]


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='llm-j', 
                    choices=['llm-j', 'emb', 'baseline'],
                    help='Method to run: llm-j (LLM Judge), emb (Embedding), baseline')
    parser.add_argument('--model', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', help='Model path')
    parser.add_argument('--judge_model', type=str, default='Qwen/Qwen2.5-7B-Instruct', help='Judge model path')
    parser.add_argument('--dataset', type=str, default='data/aime-2024.jsonl', help='Dataset path')
    parser.add_argument('--prefix', type=str, default='judge', help='Prefix')
    parser.add_argument('--start_qid', type=int, default=None, help='Start question id')
    parser.add_argument('--end_qid', type=int, default=None, help='End question id')
    parser.add_argument('--prompt_idx', type=int, default=0, help='Prompt index')
    parser.add_argument('--threshold', type=float, default=0.95, help='Prompt index')
    parser.add_argument('--allow_no_stop', action='store_true', help='Allow no stop')
    parser.add_argument('--max_workers', type=int, default=20, help='Max workers')
    parser.add_argument('--max_samples', type=int, default=1, help='Max samples')
    args = parser.parse_args()

    # Use model from command line
    model = args.model

    target_client = [OpenAI(base_url=f"http://127.0.0.1:12347/v1", api_key="None", timeout=100000)
    ,OpenAI(base_url=f"http://127.0.0.1:12348/v1", api_key="None", timeout=100000)
    ]
    if args.method == 'llm-j' or args.method == 'emb':
        draft_client = [OpenAI(base_url=f"http://127.0.0.1:12345/v1", api_key="None", timeout=100000)
        ]
        judge_client = [OpenAI(base_url=f"http://127.0.0.1:8000/v1", api_key="None", timeout=100000)
        ]


    from transformers import AutoTokenizer

    if args.method == 'llm-j' or args.method == 'emb':
        tokenizer = AutoTokenizer.from_pretrained(args.model)


    run_prefix = args.prefix + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
    output_dir = run_prefix
    # Create directory for final inputs if it doesn't exist
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.method == 'emb':
        import torch
        from sentence_transformers import SentenceTransformer

        # Set the device for the embedding model
        embedding_device = "cuda:2"
        print(f"Loading embedding model to {embedding_device}")

        # Load the sentence transformer model for embeddings
        try:
            embedding_model = SentenceTransformer("all-mpnet-base-v2", device=embedding_device)
            print(f"Successfully loaded embedding model to {embedding_device}")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            embedding_model = None

        # Define a lock for thread-safe access to the embedding model
        import threading
        embedding_lock = threading.Lock()

        def compute_similarity(sentence1, sentence2):
            """
            Compute the cosine similarity between two sentences using the embedding model.
            Uses a lock to ensure thread-safe access to the embedding model.
            
            Args:
                sentence1 (str): First sentence
                sentence2 (str): Second sentence
                
            Returns:
                float: Cosine similarity score between the two sentences (0-1 range)
            """
            if embedding_model is None:
                print("Warning: Embedding model not loaded, returning 0 similarity")
                return 0.0
                
            with embedding_lock:
                try:
                    # Encode both sentences
                    embeddings = embedding_model.encode([sentence1, sentence2])
                    
                    # Calculate cosine similarity between the embeddings
                    similarity = embedding_model.similarity(embeddings[0], embeddings[1])
                    return similarity
                except Exception as e:
                    print(f"Error computing similarity: {e}")
                    return 0.0

    # Load questions
    questions = load_questions(args.dataset)[args.start_qid:args.end_qid]
    responses = []
    # Load existing results from JSON files if they exist
    import glob
    import json
    

    temperature = 0.6
    results = []
    if args.method == 'llm-j':
        equal_prompt = equal_prompts[args.prompt_idx]
    # Get responses for each question
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_question(q, i, i0):
        #print(q.keys())
        if 'question' not in q:
            q['question'] = q['problem']
        if 'id' not in q:
            q['id'] = i0

        inp = '<｜User｜>' + q['question'] + '<｜Assistant｜>'
        next_sentence = ""
        generations = []
        generation_32 = []
        generation_1_5 = []
        accepts = []
        equals = []
        if args.method == 'llm-j' or args.method == 'emb':
            infos = []

            if args.method == 'llm-j':
                iix = len(questions) * i + i0

            token_length = 0
            while True: 
                inp = inp + next_sentence
                if args.method == 'llm-j':
                    sentence32B, finish_reason, stop_reason = get_model_response(inp, temperature=temperature, max_tokens=100, stop=['\n\n'], client=target_client[iix % 2], model=args.model, method=args.method)
                elif args.method == 'emb':
                    sentence32B, finish_reason, stop_reason = get_model_response(inp, temperature=temperature, max_tokens=100, stop=['\n\n'], client=target_client[0], model=args.model, method=args.method)
                generation_32.append(sentence32B)

                if finish_reason == 'stop' and stop_reason != '\n\n':
                    next_sentence = sentence32B
                    generations.append(next_sentence)
                    break
                sentence1_5, finish_reason1_5, stop_reason1_5 = get_model_response(inp, temperature=temperature, max_tokens=100, stop=['\n\n'], client=draft_client[0], model='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', method=args.method)
                generation_1_5.append(sentence1_5)




                if args.method == 'llm-j':
                    if args.prompt_idx == -1:
                        equal, _, _ = get_model_response(equal_prompt.format(sentence32B.strip(), sentence1_5.strip()), temperature=0.0, max_tokens=1000, client=judge_client[0], model=args.judge_model)

                        print('xEqual: ', equal)
                        if '[aligned]' in equal and '[unaligned]' not in equal and finish_reason == 'stop' and finish_reason1_5 == 'stop':
                            next_sentence = sentence1_5 
                            if finish_reason1_5 == 'stop' and stop_reason1_5 == '\n\n':
                                next_sentence += '\n\n'
                            accepts.append(1)
                        else:
                            next_sentence = sentence32B
                            if finish_reason == 'stop' and stop_reason == '\n\n':
                                next_sentence += '\n\n'
                            accepts.append(0)

                    else:
                        equal, _, _ = get_model_response(equal_prompt.format(sentence32B.strip(), sentence1_5.strip()), temperature=0.0, max_tokens=1, client=judge_client[0], model=args.judge_model)
                        print('Equal: ', equal)
                        if 'ali' in equal and 'un' not in equal and (args.allow_no_stop or (finish_reason == 'stop' and stop_reason == '\n\n')):
                            next_sentence = sentence1_5 
                            if finish_reason1_5 == 'stop' and stop_reason1_5 == '\n\n':
                                next_sentence += '\n\n'
                            accepts.append(1)
                        else:
                            next_sentence = sentence32B
                            if finish_reason == 'stop' and stop_reason == '\n\n':
                                next_sentence += '\n\n'
                            accepts.append(0)

                elif args.method == 'emb':
                    equal = compute_similarity(sentence32B.strip(), sentence1_5.strip()).item()
                    print('Equal: ', equal)
                    if equal > args.threshold and (args.allow_no_stop or (finish_reason == 'stop' and stop_reason == '\n\n')):
                        next_sentence = sentence1_5 
                        if finish_reason1_5 == 'stop' and stop_reason1_5 == '\n\n':
                            next_sentence += '\n\n'
                        accepts.append(1)
                    else:
                        next_sentence = sentence32B
                        if finish_reason == 'stop' and stop_reason == '\n\n':
                            next_sentence += '\n\n'
                        accepts.append(0)

                generations.append(next_sentence)
                infos.append((equal, generation_32[-1], generation_1_5[-1]))
                equals.append(equal)

                tokens = tokenizer.encode(next_sentence, add_special_tokens=False)
                token_length += len(tokens)
            
                if token_length > 32768:
                    print(f"Token length ({token_length}) exceeds 16000, truncating input")
                    break

        elif args.method == 'baseline':
            next_sentence, finish_reason, stop_reason = get_model_response(inp, temperature=temperature, max_tokens=32000, client=target_client[i0 % len(target_client)], model=args.model)
        
        inp = inp + next_sentence
        print('Done ', i0)
        
        # Save final input and answer to JSON file with run prefix
        if args.method == 'llm-j' or args.method == 'emb':
            output_data = {
                'question': q['question'],
                'final_input': inp,
                'answer': inp,
                'accepts': accepts,
                'equals': equals,
                'generations_32': generation_32,
                'generations_1_5': generation_1_5,
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
                'generations_32': generation_32,
                'generations_1_5': generation_1_5,
                'generations': generations,
                'gold': q['answer'],
            }
        with open(output_dir + '/' + str(q["id"]) + '_' + str(i) + '.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        print('Question: ', q['id'], 'answer:', extract_answer(inp, 'aime'), 'gold:', q['answer'], 'Spec: ', inp is None)
        return {'answer': extract_answer(inp, 'aime'), 'gold': q['answer']}
    

    # Process questions in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for i in range(args.max_samples):
            print('Running question', i, len(questions))
            future_to_question = {executor.submit(process_question, q, i, i0): q for i0, q in enumerate(questions)}
            
        for future in tqdm(as_completed(future_to_question), total=len(questions) * args.max_samples):
            question = future_to_question[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Question processing failed: {e}")
    
    correct = 0
    for result in results:
        print(result['answer'], result['gold'])
        if math_equal(str(result['answer']), str(result['gold'])):
            correct += 1
    print(f"Accuracy: {correct / len(results)}")
    accuracy_data = {
        'accuracy': correct / len(questions),
        'results': len(results)
    }
    with open(output_dir + '/'  + 'accuracy.json', 'w') as f:
        json.dump(accuracy_data, f, indent=2)

    # Save results to CSV file with run prefix
    with open(output_dir + '/' + 'results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['answer', 'gold'])
        writer.writeheader()
        for result in results:
            writer.writerow(result)

if __name__ == "__main__":
    main()
