import json
import asyncio
import os
import time
from typing import List, Dict, Any
from lr_tree import TreeNode

equal_prompt = '''<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nEvaluate whether the following two reasoning steps (s1 and s2) convey exactly the same meaning. Focus on semantic similarity rather than exact wording. 

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


async def get_model_response(prompt, model="gpt-4-turbo-preview", temperature=0.0, max_tokens=1000, stop=None, client=None):
    """Get response from OpenAI API"""
    try:
        response = await client.completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            top_p=0.95
        )
        return response.choices[0].text
    except Exception as e:
        print(f"Error getting response: {e} {model}")
        return None

async def text_accept(response1, response2, last_ending, judge_client, judge_model):
    if not last_ending:
        return False

    # Check if the responses are exactly the same
    if response1.replace(" ", "").replace("\n", "").lower() == response2.replace(" ", "").replace("\n", "").lower(): 
        return True

    prompt = equal_prompt.format(response1.strip(), response2.strip())

    response = await get_model_response(prompt, client=judge_client, model=judge_model, max_tokens=1)

    return 'ali' in response.lower() and 'un' not in response.lower() 

async def accept_func(response1, response2, last_ending, judge_client, judge_model):
    return await text_accept(response1, response2, last_ending, judge_client, judge_model)

def token_transform(token_ids, tokenizer_source, tokenizer_target):
    """Transform token IDs from source tokenizer to target tokenizer"""
    text = tokenizer_source.decode(token_ids)
    return tokenizer_target.encode(text, add_special_tokens=False)

async def run_problem(question, i, target_model, draft_model, \
                      target_tokenizer, draft_tokenizer, judge_client, \
                        target_config, draft_config, output_dir, \
                        use_spec, width, max_depth, ignore_half_sentence):
    """Run a single problem with the target and draft models"""
    question_id = i
    question_text = question['question']
    gold_answer = question['answer']

    
    # Prepare the prompt
    prompt = question_text + "\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n"

    target_prompt = target_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
        )

    draft_prompt = draft_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
        )


    target_token_ids = target_tokenizer.encode(target_prompt, add_special_tokens=False)
    draft_token_ids = draft_tokenizer.encode(draft_prompt, add_special_tokens=False)
    # response = await target_model.target(prompt_token_ids)

    if target_config['name'] == draft_config['name']:
        target2draft = lambda x: x 
        draft2target = lambda x: x
    else:
        target2draft = lambda x: token_transform(x, target_tokenizer, draft_tokenizer)
        draft2target = lambda x: token_transform(x, draft_tokenizer, target_tokenizer)

    print('Running question:', question_id, 
          'Draft prompt:', [draft_prompt], 'Target prompt:', [target_prompt],
          'Tokens: ', [target_token_ids, draft_token_ids])

    t0 = time.time()
    if use_spec:
        root = TreeNode(prefix=target_prompt, prefix_token_ids=target_token_ids, draft_prefix_token_ids=draft_token_ids, \
                       # draft_prefix=draft_prompt, draft_prefix_token_ids=draft_token_ids, \
                        width=width, idx=0, depth=1, drafter=draft_model, \
                            targeter=target_model, empty=True, max_depth=max_depth, \
                            target_config=target_config, draft_config=draft_config,\
                                qid=question_id, ignore_half_sentence=ignore_half_sentence, \
                                    accept_func=accept_func,judge_client=judge_client,draft2target=draft2target, target2draft=target2draft, judge_model=target_config['judge_model'])

        root_node = root

        while True:
            await root.traverse(root_node)
            await root.traverse_collect_main()
            await root.travel_set_accepted()
            found_eos = False
            while root.main_data is not None and root.check_judge_children():
                print('Main data: ', root.depth, root.idx)
                if any(eos_id in root.main_data['i'] for eos_id in target_config['eos_id']) or \
                   any(eos_id in root.drafter_data['i'] for eos_id in draft_config['eos_id']):
                    print('Found EOS in main data:', root.depth, root.idx, root.main_data, root.drafter_data)
                    found_eos = True
                    break

                total_tokens = root.generated_tokens + root.main_data['n']

                if total_tokens > target_config['max_tokens']:
                    found_eos = True
                    break

                children = root.children
                accepted_id = -1



                for cid, child in enumerate(children):
                    if child is None:
                        continue
                    if child.drafter_data is not None:
                        accepted = child.drafter_data['j'].result() #accept(main_data[0], child.drafter_data[0])
                        if accepted:
                            print('Accepted child...', child.depth, child.idx, [root.main_data['t'], child.drafter_data['t']])
                            accepted_id = cid
                            break
                        else:
                            child.cancel()
                            root.canceled.append(child)
                            children[cid] = None
                            del child

                if accepted_id != -1:
                    for cid, child in enumerate(children):
                        if cid == accepted_id:
                            continue
                        if child is not None:
                            child.cancel()
                            root.canceled.append(child)
                            children[cid] = None
                            del child
                    
                    root = children[accepted_id]
                    root.accepted = True
                    root_node = root
                    print('Processing ', root.depth, root.idx, root.main_data, [root.drafter_data, root.main_data])
                    
                else:
                    for cid, child in enumerate(children):
                        if child is not None:
                            child.cancel()
                            root.canceled.append(child)
                            children[cid] = None
                            del child

                    old_root = root
                    root = TreeNode(prefix=root.prefix + root.drafter_data['t'] + root.main_data['t'], \
                                    prefix_token_ids=root.prefix_token_ids + root.drafter_data['ti'] + root.main_data['i'], 
                                    draft_prefix_token_ids=root.draft_prefix_token_ids + root.drafter_data['i'] + root.main_data['di'],\
                                        width=root.width, idx=root.idx, depth=root.depth + 1, \
                                            drafter=root.drafter, targeter=root.targeter, empty=True, max_depth=root.max_depth, \
                                                generated_tokens=root.generated_tokens + root.drafter_data['n'] + root.main_data['n'], \
                                                target_config=target_config, draft_config=draft_config,\
                                                    qid=question_id, ignore_half_sentence=ignore_half_sentence, \
                                                        accept_func=accept_func, judge_client=judge_client, draft2target=draft2target, target2draft=target2draft, judge_model=target_config['judge_model'])
                    old_root.children.append(root)
                    root_node = root

                    print('Failed to find a child, creating a new root...', root.depth, root.idx, root.main_data)
                    break
            await asyncio.sleep(0.0)
            #times_step.append((t3 - t2, t2 - t1, t1 - t0))
            if found_eos:
                break

        #target_response = root.main.prefix + root.main_data['t'] if root.main_data is not None else root.prefix

        full_tokens = root.main.prefix_token_ids + root.main_data['i'] #- len(tokenizer.encode(prompt))
        full_text = target_tokenizer.decode(full_tokens, skip_special_tokens=False)
        generation_tokens = full_tokens[len(target_token_ids):]
        
        
    else:
        # Run the target model
        response = await target_model.target(target_token_ids, max_tokens=target_config['max_tokens'],
                                              temperature=target_config['temperature'],
                                              top_p=target_config['top_p'], top_k=target_config['top_k'],
                                              stop=[])
        generation_text = response[0]
        full_text = target_prompt + generation_text
        generation_tokens = response[-1]
        full_tokens = target_token_ids + generation_tokens

    t1 = time.time()

    print('Finished: ', len(generation_tokens), t1 - t0, 'Speed: ', len(generation_tokens) / (t1 - t0) , 'tokens/s')

    # Save the results
    result = {
        'question_id': question_id,
        'question': question_text,
        'draft_prompt': draft_prompt,
        'target_prompt': target_prompt,
        'generation_tokens': generation_tokens,
        'generation_text': target_tokenizer.decode(generation_tokens, skip_special_tokens=False),
        'full_text': full_text,
        'full_tokens': full_tokens,
        'gold': gold_answer,
        'time_taken': t1 - t0,
        'speed': len(generation_tokens) / (t1 - t0),
        'draft_config': draft_config,
        'target_config': target_config,
    }

    output_file = os.path.join(output_dir, f"{question_id}.json")
    with open(output_file, 'w') as f:
        json.dump(result, f)
