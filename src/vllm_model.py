from vllm.inputs import TokensPrompt
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import PromptType
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
import asyncio
import time
from transformers import AutoTokenizer

class LLMModel:
    def __init__(self, model,  eos=None, eos_id=None, gpu_ids="", enable_n_gram=False,  vllm_config={}):
        self.gpu_ids = gpu_ids
        self.eos_id = eos_id

        if enable_n_gram:
            print('Ngram Setting Up Model: ', [model, gpu_ids])
            engine_args = AsyncEngineArgs(
                model=model,
                enforce_eager=vllm_config['force_eager'],
                tensor_parallel_size=len(gpu_ids.split(',')),
                data_parallel_size=1,
                gpu_memory_utilization=0.7,
                enable_prefix_caching=True,
                enable_chunked_prefill=True,
                max_num_batched_tokens=8192,
                speculative_config={
                    "method": "ngram",
                    "num_speculative_tokens": vllm_config['num_speculative_tokens'],
                    "prompt_lookup_max": vllm_config['prompt_lookup_max'],
                },
            )
        else:
            print('Setting Up Model: ', [model, gpu_ids])

            engine_args = AsyncEngineArgs(
                model=model,
                enforce_eager=vllm_config['force_eager'],
                tensor_parallel_size=len(gpu_ids.split(',')),
                data_parallel_size=1,
                gpu_memory_utilization=0.7,
                enable_prefix_caching=True,
                enable_chunked_prefill=True,
                max_num_batched_tokens=8192,
            )
    
        self.prefix = model.split('/')[-1].split('-')[0]  # Extract prefix from model name
        self.engine = AsyncLLM.from_engine_args(engine_args)
        self.request_id = 1
        self.eos = eos
        self.enable_n_gram = enable_n_gram
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    async def generate(self,
                   prompt: PromptType, max_tokens: int, temperature, top_p, top_k, stop) -> tuple[int, str]:
        # Ensure generate doesn't complete too fast for cancellation test.
        #await asyncio.sleep(0.00001)
        prompt = TokensPrompt(prompt_token_ids=prompt)

        sampling_params = SamplingParams(max_tokens=max_tokens,
                        ignore_eos=False,
                        output_kind=RequestOutputKind.FINAL_ONLY,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop,
                        top_k=top_k, seed=int(time.time()))
        self.request_id += 1
        async for out in self.engine.generate(request_id=self.prefix+f"request-{self.request_id}",
                                        prompt=prompt,
                                        sampling_params=sampling_params):


            await asyncio.sleep(0.)
        
        #print('Output: ', out.outputs[0])
        out.outputs[0].token_ids = list(out.outputs[0].token_ids)
        num_tokens = len(out.outputs[0].token_ids)

        if out.outputs[0].finish_reason == 'stop':
            if out.outputs[0].stop_reason is not None:
                if out.outputs[0].stop_reason == self.eos_id:
                    return out.outputs[0].text + '<|endoftext|>', out.outputs[0].finish_reason, out.outputs[0].stop_reason, num_tokens, out.outputs[0].token_ids
                
                #print('Stopping: ', [out.outputs[0].text, out.outputs[0].stop_reason,
                #                     tokenizer.decode(out.outputs[0].token_ids), tokenizer.encode(out.outputs[0].text + out.outputs[0].stop_reason)], self.eos_id)
                #if '\n\n' in out.outputs[0].text:
                #    print('Double Newline Stopping: ', [out.outputs[0].text, out.outputs[0].stop_reason], self.eos_id)
                #    exit(0)
                if self.enable_n_gram:
                    out.outputs[0].token_ids = self.tokenizer.encode(out.outputs[0].text + out.outputs[0].stop_reason)
                return out.outputs[0].text + out.outputs[0].stop_reason, out.outputs[0].finish_reason, out.outputs[0].stop_reason, num_tokens, out.outputs[0].token_ids
            else:
                print('Finishing: ', out.outputs[0].finish_reason, out.outputs[0].stop_reason, self.eos_id)
                return out.outputs[0].text, out.outputs[0].finish_reason, self.eos, num_tokens, out.outputs[0].token_ids
        else:
            return out.outputs[0].text, out.outputs[0].finish_reason, out.outputs[0].stop_reason, num_tokens, out.outputs[0].token_ids

### Drafter model
class Drafter:
    def __init__(self, model, eos=None, eos_id=None, draft_gpu_id=None, \
                 enable_n_gram=False, vllm_config={'force_eager': False, 'num_speculative_tokens': 4, 'prompt_lookup_max': 2}):
        print('Drafting')
        self.model = LLMModel(model, eos, eos_id, draft_gpu_id, enable_n_gram, vllm_config)
        
    def draft(self, prompt, max_tokens=100, temperature=0.6, top_p=0.95, top_k=20, stop=['\n\n']):
        return self.model.generate(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, stop=stop)

### Target Model 
class Targeter:
    def __init__(self, model, eos=None, eos_id=None, target_gpu_id=None, \
                 enable_n_gram=False, vllm_config={'force_eager': False, 'num_speculative_tokens': 4, 'prompt_lookup_max': 2}):
        print('Targeting')
        self.model = LLMModel(model, eos, eos_id, target_gpu_id, enable_n_gram, vllm_config)
        
    def target(self, prompt, max_tokens=100, temperature=0.6, top_p=0.95, top_k=20, stop=['\n\n']):
        return self.model.generate(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, stop=stop)
