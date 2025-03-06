from utils import read_file

# path
seed_path = 'data/evolve/seed_48k.json'
save_path = 'data/evolve/evolved.json'
final_data_path = 'data/evolve/final.json'
deep_cache = 'cache/d[].json'
fuse_cache = 'cache/f[].json'
round_info = 'cache/round_info.json'
comm_to_cal = 'cache/to_cal/'
comm_gen_resp = 'cache/gen_resp/'
comm_cal_done = 'cache/cal_done/'

# model_path = "/search/ai/jiuding/LLM_ZOO/llama3/Meta-Llama-3-8B-Instruct/"
model_path = "/cfs/cfs-g22qkwzd/jiuding/LLM_ZOO/llama3-instruct"
cal_cache = "cache/amb/"

# config
target_deep = 8000
target_fuse = 8000
max_len_to_end = 4000
target_round = 6

gpu_id = "0,1,2,3"
gpu_id = gpu_id.split(',')
max_len = 4096

# prefix
prefix_evol_deep = read_file('prefix/evol/difficulty.txt')
prefix_evol_fuse = read_file('prefix/evol/task_fusion.txt')
prefix_resp = read_file('prefix/get_resp.txt')
replace_key_deep = {'{{  prompt  }}': 'seed_inst', "{{  extracted  }}": 'seed_extracted'}
replace_key_fuse = {'{{  prompt1  }}': 'p1', "{{  extracted1  }}": 'extracted1',
                    '{{  prompt2  }}': 'p2', "{{  extracted2  }}": 'extracted2'}
replace_get_resp = {'{{  prompt  }}': 'instruction'}

# data crawling settings
model_name = 'gpt-4o'
num_workers = 20

prefix = None
replace_dict = None
cache_file_path = None
response_store_key = None

