import asyncio
import datetime
import json
import os
import time

class MainNode:
    def __init__(self, targeter=None, target_config=None, target2draft=None):
        self.targeter = targeter
        self.task = None
        self.last_ending = None
        self.target_config = target_config
        self.target2draft = target2draft

    def target(self, prefix_token_ids=None, last_ending=None):
        self.last_ending = last_ending
        self.prefix_token_ids = prefix_token_ids
        self.task = asyncio.create_task(self.targeter.target(prefix_token_ids, temperature=self.target_config['temperature'], \
                                                      top_p=self.target_config['top_p'], max_tokens=self.target_config['step_tokens'], \
                                                      stop=self.target_config['stop'], top_k=self.target_config['top_k']))

    def done(self):
        return self.task.done()
    
    async def materialize(self):
        response, stop_reason, matched_stop, num_tokens, token_ids = await self.task
        return {'t': response, 'r': stop_reason, 's': matched_stop, 'n': num_tokens, 
                'i': token_ids, 'l': self.last_ending, 'di': self.target2draft(token_ids)}
    
    def cancel(self):
        if self.task is not None:
            self.task.cancel()
        
class DrafterNode:
    def __init__(self, draft_prefix_token_ids=None, drafter=None, draft_config=None, draft2target=None):
        self.drafter = drafter
        self.task = None
        self.draft_config = draft_config
        self.draft_prefix_token_ids = draft_prefix_token_ids
        self.draft2target = draft2target
    
    def draft(self):
        if self.drafter == 'empty':
            return
        self.task = asyncio.create_task(self.drafter.draft(self.draft_prefix_token_ids, temperature=self.draft_config['temperature'], \
                                                      top_p=self.draft_config['top_p'], max_tokens=self.draft_config['step_tokens'], \
                                                      stop=self.draft_config['stop'], top_k=self.draft_config['top_k'])) 
    

    def done(self):
        if self.drafter == 'empty':
            return True        
        return self.task.done()
    
    async def materialize(self):
        if self.drafter == 'empty':
            return {'t': '', 'r': None, 's': None, 'n': 0, 'i': [], 'ti':[], 'j': None}
        response, stop_reason, matched_stop, num_tokens, token_ids = await self.task
        return {'t': response, 'r': stop_reason, 's': matched_stop, 'n': num_tokens, 'i': token_ids, 'ti':self.draft2target(token_ids), 'j': None}
        # return response, stop_reason, matched_stop, num_tokens, token_ids, None
    
    def cancel(self):
        if self.drafter == 'empty':
            return
        self.task.cancel()
        
#draft_prefix=None, draft_prefix_token_ids=None,

class TreeNode:
    def __init__(self, prefix=None, prefix_token_ids=None, draft_prefix_token_ids=None, width=3, idx=0, depth=1, drafter=None, targeter=None, empty=False, \
                 max_depth=None, generated_tokens=0, target_config=None,draft_config=None, qid=None, ignore_half_sentence=True, 
                 accept_func=None, judge_client=None, draft2target=None, target2draft=None, build_info=None, judge_model=None):
        self.prefix = prefix
        self.prefix_token_ids = prefix_token_ids
        self.draft_prefix_token_ids = draft_prefix_token_ids
        #self.draft_prefix = draft_prefix
        #self.draft_prefix_token_ids = draft_prefix_token_ids
        
        self.width = width
        self.idx = idx
        self.depth = depth
        self.empty = empty
        self.accepted = False
        self.max_depth = max_depth
        self.drafter = drafter
        self.targeter = targeter
        self.children = []
        self.canceled = []
        self.base = DrafterNode(draft_prefix_token_ids=draft_prefix_token_ids, drafter=drafter if not empty else 'empty', draft_config=draft_config, draft2target=draft2target)
        self.main = MainNode(targeter=targeter, target_config=target_config, target2draft=target2draft)
        self.draft()
        self.qid = qid
        self.ignore_half_sentence = ignore_half_sentence
        self.accept_func = accept_func
        self.judge_client = judge_client
        self.judge_model = judge_model
        self.drafter_data = None
        self.main_data = None
        self.generated_tokens = generated_tokens
        self.target_config=target_config
        self.draft_config=draft_config 
        self.draft2target = draft2target
        self.target2draft = target2draft
    
    def draft(self):
        self.base.draft()

    def main_not_started(self):
        return self.main.task is None

    def main_started(self):
        return self.main.task is not None

    def target(self):
        self.main.target(self.prefix_token_ids + self.drafter_data['ti'],\
                            (self.prefix + self.drafter_data['t']).endswith("\n\n") or not self.ignore_half_sentence)
    
    def check_base(self):
        return self.base.done()
    
    def check_main(self):
        return self.main.done()

    def draft_materialize(self):
        return self.base.materialize()
    
    def target_materialize(self):
        return self.main.materialize()

    def cancel(self):
        self.base.cancel()
        self.main.cancel()
        for child in self.children:
            child.cancel()
            self.canceled.append(child)

    async def start_main_if_possible(self):
        if self.check_base() and self.main_not_started():
            self.drafter_data = await self.draft_materialize()
            self.target()
            return True
        return False

    def allocate_children(self, root_node):
        if self.check_base() and len(self.children) == 0 and (self.depth - root_node.depth) < self.max_depth:
            for i in range(self.width):
                self.children.append(TreeNode(prefix=self.prefix + self.drafter_data['t'], \
                                              prefix_token_ids=self.prefix_token_ids + self.drafter_data['ti'],
                                                draft_prefix_token_ids=self.draft_prefix_token_ids + self.drafter_data['i'],\
                                                width=self.width, idx=i, depth=self.depth + 1, drafter=self.drafter, targeter=self.targeter, \
                                                    empty=False, max_depth=self.max_depth, generated_tokens=self.generated_tokens + self.drafter_data['n'], \
                                                    target_config=self.target_config, draft_config=self.draft_config,\
                                                        qid=self.qid, ignore_half_sentence=self.ignore_half_sentence, accept_func=self.accept_func, judge_client=self.judge_client,\
                                                        draft2target=self.draft2target, target2draft=self.target2draft))
            return True
        return False

    async def collect_main_if_possible(self):
        if self.main_started() and self.check_main() and self.main_data is None:
            self.main_data = await self.target_materialize() #text, finish_reason, stop_reason, num_tokens, token_ids, last_ending = await self.main_materialize()
            return True
        return False
    
    async def travel_set_accepted(self):
        if self.main_data is not None:
            for child in self.children:
                if child is not None and child.drafter_data is not None and child.drafter_data['j'] is None:
                    child.drafter_data['j'] = asyncio.create_task(self.accept_func(self.main_data['t'], child.drafter_data['t'], self.main_data['l'], self.judge_client, self.judge_model))

        for child in self.children:
            await child.travel_set_accepted()

    def check_judge_children(self):
        for child in self.children:
            if child is not None and child.drafter_data is not None:
                if child.drafter_data['j'] is None or not child.drafter_data['j'].done():
                    return False
        return True
    
    async def traverse(self, root_node):
        if await self.start_main_if_possible(): 
            pass
            
        if self.allocate_children(root_node):
            pass

        for child in self.children:
            await child.traverse(root_node)
    
    async def traverse_collect_main(self):
        if await self.collect_main_if_possible():
            pass
        for child in self.children:
            await child.traverse_collect_main()

