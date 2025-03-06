from utils import load_data, save_data, check_file, sort_list_and_return_indices, crawl_with_instruct
import config as cfg_ori
import tiktoken
import numpy as np
from copy import deepcopy
import torch
import time

tikto = tiktoken.get_encoding("cl100k_base")

replace_dict_deep = {
    '**Background Settings:**': ["### Background Settings:", "### Merged Background Settings:",
                                 "### Additional Background Settings:", "## Background Settings:",
                                 "## Merged Background Settings:",
                                 "## Additional Background Settings:",
                                 "**Background Setting:**",
                                 '**Additional Background Settings:**',
                                 "**Merged Background Settings:**"],
    '**Objectives:**': {"### Merged Objectives:", "### Objectives:", "### Additional Objectives:",
                        "## Merged Objectives:", "## Objectives:", "## Additional Objectives:",
                        '**Merged Objectives:**',
                        '**Additional Objectives:**',
                        "**Objective:**"},
    '**Constraints:**': ["### Constraints:", "### Merged Constraints:", "### Additional Constraints:",
                         "## Constraints:", "## Merged Constraints:", "## Additional Constraints:",
                         '**Merged Constraints:**',
                         '**Constraint:**',
                         "**Additional Constraints:**"],
    " Constraints:**": [' Constraint:**'],
    " Objectives:**": [' Objective:**'],
    " Settings:**": [' Setting:**'],
}
replace_dict_fuse = {
    "**Fused Prompt:**": ["### Fused Prompt:", "## Fused Prompt:", "### Merged Prompt:", "## Merged Prompt:",
                          "**Fused Prompt**:", '**New Prompt:**'],
    "**Fused Constraints:**": ["### Fused Constraints:", "## Fused Constraints:", "### Merged Constraints:",
                               "## Merged Constraints:",
                               "**Fused Constraints**:", '**Merged Constraints:**'],
    "**Fused Objectives:**": ["### Fused Objectives:", "## Fused Objectives:", "### Merged Objectives:",
                              "## Merged Objectives:",
                              "**Fused Objectives**:", '**Merged Objectives:**'],
    "**Fused Background Settings:**": ["### Fused Background Settings:", "## Fused Background Settings:",
                                       "### Merged Background Settings:", "## Merged Background Settings:",
                                       "**Fused Background Settings**:", '**Merged Background Settings:**'],
    " Constraints:**": [' Constraint:**'],
    " Objectives:**": [' Objective:**'],
    " Settings:**": [' Setting:**'],
    "**Prompt:**": ['### Prompt:']
}


def read_cache(path):
    try:
        data = load_data(path)
        print(f"{len(data)} cached sample loaded from: {path}")
        return data
    except:
        with open(path, 'w') as f:
            pass
        print(f"Create cache file {path}")
        return []


class evol(object):
    def __init__(self, cfg=cfg_ori):
        self.to_evolve = None
        self.cfg = cfg
        self.pool = {}
        self.round_info = {}
        self.to_calcuate = []
        self.to_wait = []
        self.current_round = -1
        # round_info --> {1:[sample list deep, sample list fuse, complete?], 2:...}
        self.round_info = read_cache(self.cfg.round_info)
        if not self.round_info:
            self.round_info = {'-1': [[], [], False]}
        try:
            self.pool = load_data(cfg.save_path)
            print(f"{len(self.pool)} samples loaded into pool from {self.cfg.save_path}")
        except:
            print(f"No cache pool found.")
            seed = load_data(cfg.seed_path)
            pool = []
            for key in seed:
                for sample in seed[key]:
                    new_sample = {
                        'id': sample['id'],
                        'instruction': sample['instruction'],
                        'response': sample['response'],
                        'extracted': sample['extracted'],
                        "evol_info": {"level": 0,
                                      "uncertainty": {'self': [], 'gpt': [], 'self_response': None},
                                      "obj_count": 1, "from": sample['evol_info']['from'],
                                      "method": sample['evol_info']['method'], 'deep_to': None, 'fuse_to': []}}

                    pool.append(new_sample)
            self.cache_to_cal(pool, 0, 'seed')
            self.collect_uncertainty()

        self.task_counter = {'ShareGPT': 0, 'Alpaca': 0, 'MATH': 0, 'gsm8k': 0, 'CodeAlpaca': 0}
        self.update_status()

    def update_resp_len(self):
        for id, sample in self.pool.items():
            if 'resp_size' not in sample:
                sample['size'] = len(tikto.encode(sample['instruction'] + "\n" + sample['response']))
                if sample['size'] > self.cfg.max_len_to_end:
                    sample['evol_info']['deep_to'] = '**END**'
                    sample['evol_info']['fuse_to'] = '**END**'

    def cal_fuse_weight(self, sample):
        # skip all instructions with over length response
        if sample['evol_info']['deep_to'] == '**END**': return 0
        # if sample['evol_info']['fuse_to'] == '**FAILED**': return 0
        sample_id = sample['id']
        task = sample_id.split("|")[0]
        task_count = self.task_counter[task] + 1
        sample_count = len(sample['evol_info']['fuse_to']) + 1
        objective_count = sample['evol_info']['obj_count']
        uncertainty = sample['evol_info']['uncertainty']['self'][1]
        return 1 / (task_count * sample_count * objective_count) ** 2 / (uncertainty + 1e-6)

    def sample_deep(self, target=None):
        if target is None:
            target = self.cfg.target_deep
        score_all = []
        id_all = []
        for id, sample in self.pool.items():
            id_all.append(id)
            score = sample['evol_info']['uncertainty']['self']
            score_gpt = sample['evol_info']['uncertainty']['gpt']
            if sample['evol_info']['deep_to'] is None:
                score_all.append(score[1] + score_gpt[0])
            else:
                score_all.append(0)

        index_sorted_score = sort_list_and_return_indices(score_all)

        return [id_all[index] for index in index_sorted_score[-target:]]

    def sample_fuse(self, target=None):
        if target is None:
            target = self.cfg.target_fuse
        # prepare in-domain sample
        target_in = int(target / 2)
        target_cross = target - target_in
        # use set to prevent duplication
        sampled_in = set()
        sampled_cross = set()
        # extract all sample ids
        key_list = list(self.pool.keys())
        # calculate and normalize weights
        weight = np.array([self.cal_fuse_weight(sample) for id, sample in self.pool.items()])
        normalized_weights = weight / sum(weight)
        # sample first sample id of each pair
        sampled_indices_first = np.random.choice(len(key_list), target, replace=False, p=normalized_weights)
        # change weights to 0 for sampled ids
        sample_key_first = [key_list[idx] for idx in sampled_indices_first]
        weight[sampled_indices_first] = 0
        normalized_weights = weight / sum(weight)
        # sample the second sample id of each pair
        p = 0  # pointer which point to the first id of each pair
        while len(sampled_in) < target_in or len(sampled_cross) < target_cross:
            remain = target - len(sampled_in) - len(sampled_cross)
            sampled_indices_second = np.random.choice(len(key_list), remain, replace=False, p=normalized_weights)
            for idx in sampled_indices_second:
                id1 = sample_key_first[p]
                id2 = key_list[idx]
                if id1.split("|")[0] == id2.split("|")[0] and len(sampled_in) < target_in:
                    pair = tuple(sorted([id1, id2]))
                    if pair not in sampled_in:
                        sampled_in.add(pair)
                        p += 1
                elif id1.split("|")[0] != id2.split("|")[0] and len(sampled_cross) < target_cross:
                    pair = tuple(sorted([id1, id2]))
                    if pair not in sampled_cross:
                        sampled_cross.add(pair)
                        p += 1
                else:
                    continue
        union = sampled_in.union(sampled_cross)
        return list(union)

    def evolve(self):
        print('Evolving')
        for r in range(1, self.cfg.target_round + 1):
            r = str(r)
            if r in self.round_info:
                candidate_key_deep, candidate_key_fuse, complete = self.round_info[r]
                if complete:
                    print(f"Round {r} is finished, skipping...")
                    continue
                print(f"Continuing round {r}...")
                self.current_round = r
            else:
                print(f"Starting from round {r}...")
                self.current_round = r

                print('Sampling for depth evolve...')
                candidate_key_deep = self.sample_deep()

                print('Sampling for fuse evolve...')
                candidate_key_fuse = self.sample_fuse()
                self.round_info[r] = [candidate_key_deep, candidate_key_fuse, False]
                self.save_round()

            candidate_sample_deep = [{
                'id': self.pool[key]['id'] + f"|d{r}",
                'instruction': None,
                'response': None,
                'evol_info': {'level': r,
                              "uncertainty": {'self': [], 'gpt': [], 'self_response': None},
                              "obj_count": self.pool[key]['evol_info']['obj_count'],
                              'from': self.pool[key]['id'],
                              'method': f'difficulty', 'deep_to': None, 'fuse_to': []},
                'extracted': None,
                'seed_inst': self.pool[key]['instruction'],
                'seed_extracted': self.pool[key]['extracted'],
            } for key in candidate_key_deep]

            candidate_sample_fuse = [{
                'id': f"{self.pool[k1]['id']}*{r}*{self.pool[k2]['id']}",
                'instruction': None,
                'response': None,
                "evol_info": {
                    "level": r,
                    "uncertainty": {'self': [], 'gpt': [], 'self_response': None},
                    "obj_count": self.pool[k1]['evol_info']['obj_count'] + self.pool[k1]['evol_info']['obj_count'],
                    "from": (self.pool[k1]['id'], self.pool[k2]['id']),
                    "method": f"fusion|{'C' if self.pool[k1]['id'].split('|')[0] != self.pool[k2]['id'].split('|')[0] else 'I'}",
                    'deep_to': None, 'fuse_to': []},
                'extracted': None,
                'p1': self.pool[k1]['instruction'],
                'p2': self.pool[k2]['instruction'],
                "extracted1": self.pool[k1]['extracted'],
                "extracted2": self.pool[k2]['extracted'],
            } for k1, k2 in candidate_key_fuse]
            self.to_evolve = [candidate_sample_deep, candidate_sample_fuse]
            print(f"datasize| depth {len(candidate_sample_deep)} | fuse {len(candidate_sample_fuse)}")
            self.gpt_evol_deep()
            self.gpt_evol_fuse()
            self.collect_uncertainty()
        #     collect all data
        data = []
        for key, sample in self.pool.items():
            data.append({
                'id': sample['id'],
                'instruction': sample['instruction'],
                'response': sample['response']
            })
        save_data(data, self.cfg.final_data_path)

    def gpt_evol_fuse(self, ):
        r = self.current_round
        print(f"Round {r} fuse evolving ...")
        # update cfg
        self.cfg.prefix = self.cfg.prefix_evol_fuse
        self.cfg.cache_file_path = self.cfg.fuse_cache.replace('[]', str(r))
        self.cfg.replace_dict = self.cfg.replace_key_fuse
        self.cfg.response_store_key = 'fuse_evol'
        result = crawl_with_instruct(self.to_evolve[1], self.cfg)
        print("Collecting evolved instructions")
        evolved = []

        for sample in result:
            fuse = sample[self.cfg.response_store_key]
            for to_replace, pattern in replace_dict_fuse.items():
                for p in pattern:
                    fuse = fuse.replace(p, to_replace)
            if '**Fused Prompt:**' not in fuse:
                print('Failed Fusion')
                # for sample_id in sample['evol_info']['from']:
                #     self.pool[sample_id]['evol_info']['fuse_to'] = "**FAILED**"
                continue
            else:
                sample['extracted'] = fuse[:fuse.index('**Fused Prompt:**')].replace('**Fused ', '**')
                sample['instruction'] = fuse[fuse.index('**Fused Prompt:**') + len('**Fused Prompt:**\n'):]
                del sample[self.cfg.response_store_key]
                evolved.append(sample)
        # update cfg for getting response
        print(f"{len(evolved)} sample are successfully task evolved in round {r}. Crawling response...")
        self.cfg.prefix = self.cfg.prefix_resp
        self.cfg.cache_file_path = self.cfg.fuse_cache.replace('[]', str(r) + '_response')
        self.cfg.replace_dict = self.cfg.replace_get_resp
        self.cfg.response_store_key = 'response'

        print(f"Round {r} fuse response collecting ...")
        result = crawl_with_instruct(evolved, self.cfg)
        self.cache_to_cal(result, r, 'f')

    def gpt_evol_deep(self, ):
        r = self.current_round
        print(f"Round {r} depth evolving ...")
        # update cfg
        self.cfg.prefix = self.cfg.prefix_evol_deep
        self.cfg.cache_file_path = self.cfg.deep_cache.replace('[]', str(r))
        self.cfg.replace_dict = self.cfg.replace_key_deep
        self.cfg.response_store_key = 'difficulty_evol'
        result = crawl_with_instruct(self.to_evolve[0], self.cfg)
        print("Collecting evolved instructions")
        evolved = []

        for sample in result:
            resp = sample[self.cfg.response_store_key]
            for to_replace, pattern in replace_dict_deep.items():
                for p in pattern:
                    resp = resp.replace(p, to_replace)
            try:
                assert '**Prompt:**' in resp
                assert '**Background Settings:**' in resp
                assert '**Objectives:**' in resp
                assert '**Constraints:**' in resp
                prompt_start = resp.index('**Prompt:**') + len('**Prompt:**\n')
                prompt_end = resp.index('**Background Settings:**')
                instruction = resp[prompt_start:prompt_end]
                extracted = resp[prompt_end:]
                sample['instruction'] = instruction
                sample['extracted'] = extracted
                del sample[self.cfg.response_store_key]
                evolved.append(sample)
            except:
                print(f"----------------{sample['id']}-------------------")
                print(resp)
                #  try fix
                source = sample['evol_info']['from']
                self.pool[source]['evol_info']['deep_to'] = '**FAILED**'

        # update cfg for getting response
        print(f"{len(evolved)} sample are successfully depth evolved in round {r}. Crawling response...")
        self.cfg.prefix = self.cfg.prefix_resp
        self.cfg.cache_file_path = self.cfg.deep_cache.replace('[]', str(r) + '_response')
        self.cfg.replace_dict = self.cfg.replace_get_resp
        self.cfg.response_store_key = 'response'

        print(f"Round {r} depth response collecting ...")
        result = crawl_with_instruct(evolved, self.cfg)
        self.cache_to_cal(result, r, 'd')

    def collect_uncertainty(self):
        print("Waiting calculation...")
        for idx, path in enumerate(self.to_wait):
            while not check_file(path):
                time.sleep(10)
            print(f"{path} is found!")
        # give enough time to finish saving
        time.sleep(10)
        print("Updating Pool")
        for idx, path in enumerate(self.to_wait):
            new = load_data(path)
            for sample in new:
                self.pool[sample['id']] = sample
                # update tracking
                if sample['evol_info']['method'] in ['seed', 'task_diversification']:
                    continue
                source_id = sample['evol_info']['from']
                if type(source_id) == str:
                    self.pool[source_id]['evol_info']['deep_to'] = source_id
                elif type(source_id) == list:
                    for sid in source_id:
                        self.pool[sid]['evol_info']['fuse_to'].append([sid])
                else:
                    print(f"Unknown source_id {source_id}")

        # reset to_wait
        self.to_wait = []
        # save pool and round_info
        self.round_info[str(self.current_round)][2] = True
        self.update_status()
        self.save()

    def update_status(self):
        # update information
        self.update_resp_len()
        self.task_counter = {'ShareGPT': 0, 'Alpaca': 0, 'MATH': 0, 'gsm8k': 0, 'CodeAlpaca': 0}
        for r, (deep, fuse, complete) in self.round_info.items():
            for pair in fuse:
                for key in pair:
                    task = key.split("|")[0]
                    self.task_counter[task] += 1
                    # ensure they have same counter since they are all math
                    if task == 'gsm8k':
                        self.task_counter['MATH'] += 1
                    if task == 'MATH':
                        self.task_counter['gsm8k'] += 1
        self.to_evolve = None

    def save(self):
        print(f"Current Poll Size: {len(self.pool)}")
        save_data(self.pool, self.cfg.save_path)
        self.save_round()

    def save_round(self):
        save_data(self.round_info, self.cfg.round_info)

    def cache_to_cal(self, result, r, name):
        save_data(result, self.cfg.comm_to_cal + f"{name}{r}_to_cal.json")
        self.to_wait.append(self.cfg.comm_cal_done + f"{name}{r}_to_cal.json")
