import argparse
import pprint
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = 'spawn'

import json
import torch
from multiprocessing import set_start_method, Process
from transformers import AutoTokenizer
import subprocess
from tqdm import tqdm
import random

set_start_method('spawn', force=True)


def load_data(path):
    try:
        return json.load(open(path, 'r'))
    except:
        with open(path, 'r') as f:
            data = []
            for line in f:
                data.append(json.loads(line))
    return data


def save_data(data, path, jsonl=False):
    if jsonl:
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    else:
        json.dump(data, open(path, 'w'), indent=2)


def get_num_gpus():
    try:
        n = len(
            subprocess.check_output(['nvidia-smi', '-L']).decode('utf-8').strip().split('\n'))
    except OSError:
        n = 0
    return n


def inference_on_gpu(gpu_id, model_path, remain, prompts, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, tensor_parallel_size=1, trust_remote_code=True,
              enforce_eager=False, tokenizer_mode='auto', dtype='float16')

    generation_config = SamplingParams(temperature=0, top_p=1, max_tokens=2048)
    # split prompts into smaller batch
    spl = [prompts[i:i + 100] for i in range(0, len(prompts), 100)]
    idx = 0  # record index of current sample
    done = []
    for batch in tqdm(spl):
        # get batch response
        outputs = llm.generate(batch, generation_config, use_tqdm=False)
        # collect all response text
        out = [out.outputs[0].text for out in outputs]
        # record responses
        done += out
        # cache the results
        with open(args.cache_path, 'a') as file:
            for resp in out:
                remain[idx]['evol_info']['uncertainty']['self_response'] = resp
                file.write(json.dumps(remain[idx]) + '\n')
                idx += 1
    # torch.save((done, remain), f'response_tmp_{gpu_id}.torch')
    save_data(done, args.save_path)


def run(args):
    NUM_GPU = args.num_gpus
    print(f"There are {NUM_GPU} GPUs available!")
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # load data
    data = load_data(args.data_path)
    # load cache
    try:
        cache = load_data(args.cache_path)
        print(f"{len(cache)} samples loaded from {args.cache_path}")
    except:
        with open(args.cache_path, 'w') as f:
            pass
        cache = []
    result = []
    # collect all sample keys
    keys = {sample['id']: 0 for sample in data}
    # check all cache, collect from cache if sample is required
    for sample in cache:
        if sample['id'] in keys:
            result.append(sample)
    # collect all cache keys
    keys = {sample['id']: 0 for sample in cache}
    # collect all remaining sample in data that are not in cache
    remain = [sample for sample in data if sample['id'] not in keys]
    num_samples = len(remain)
    print("Number of Remaining Samples: {}".format(num_samples))

    # shuffle remain to ensure a similar task distribution on each GPU
    random.shuffle(remain)
    # load tokenizer and apply template
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": sample['instruction']}],
                                             tokenize=False, add_generation_prompt=True) for sample in remain]
    print(f"Total number of instructions: {len(prompts)}")

    # inference_on_gpu("0,1,2,3", args.model_path, remain, prompts, args)

    processes = []
    samples_per_gpu = len(prompts) // NUM_GPU

    for i, gpu_id in enumerate(args.gpu_id):
        start_idx = i * samples_per_gpu
        end_idx = (i + 1) * samples_per_gpu if i != NUM_GPU - 1 else len(prompts)
        p = Process(target=inference_on_gpu,
                    args=(gpu_id, args.model_path, remain[start_idx:end_idx], prompts[start_idx:end_idx], args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def update_args(args):
    if args.gpu_id == 'default':
        args.num_gpus = get_num_gpus()
        args.gpu_id = [str(i) for i in range(args.num_gpus)]
    else:
        args.gpu_id = args.gpu_id.split(',')
        args.num_gpus = len(args.gpu_id)
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/search/ai/jiuding/LLM_ZOO/llama3/Meta-Llama-3-8B-Instruct',
                        help="")
    parser.add_argument('--data_path', type=str, default='data/evolve/d1_response.json', help="")
    parser.add_argument('--save_path', type=str, default='data/amb/d1_response_[model_name].json', help="")
    parser.add_argument('--cache_path', type=str, default='data/amb/cache/d1_response_[model_name].json', help="")
    parser.add_argument('--gpu_id', type=str, default='4,5,6,7', help="")
    parser.add_argument('--max_len', type=int, default=4096, help="")
    args = parser.parse_args()
    args = update_args(args)

    run(args)
