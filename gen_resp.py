import json
import argparse
import pprint
import sys
import os
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from multiprocessing import Pool
import subprocess


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


def make_chat_prompt(task_prompt, tokenizer):
    try:
        task_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": task_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            padding=False,
            max_length=4096,
            truncation=False
        )
    except:
        task_prompt = f"<s> [INST] {task_prompt} [/INST]"
    return task_prompt


def get_model(load_8bit: bool = False, base_model: str = "bigcode/starcoder", device=None, args=None):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, )
    print(f"loading model to device:{device}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=f"cuda:{device}",
        trust_remote_code=True,
    )
    if 'qwen' in base_model:
        print('Modifying tokenizer for qwen model')
        tokenizer.bos_token_id = tokenizer.encode(
            text='<|im_start|>',
            add_special_tokens=False
        )[0]
        tokenizer.eos_token_id = tokenizer.encode(
            text='<|im_end|>',
            add_special_tokens=False
        )[0]
        print(tokenizer.bos_token_id, tokenizer.eos_token_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()

    return tokenizer, model


def chunkify(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def run(args):
    NUM_GPU = args.num_gpus
    NUM_WORKER = args.num_workers
    print(f"There are {NUM_GPU} GPUs available!")
    argsdict = vars(args)
    print(pprint.pformat(argsdict))
    data = load_data(args.data_path)
    for index, sample in enumerate(data):
        sample['id'] = index
    num_samples = len(data)
    print("Number of samples: {}".format(num_samples))

    print(f"Loaded {args.model_path}.")
    chunks = [data[i::NUM_WORKER] for i in range(NUM_WORKER)]
    GPU_index = args.gpu_id
    print(f"Seperate Data into {len(chunks)} chunks")
    chunk_data = [(args, index, chunk, GPU_index[index]) for index, chunk in enumerate(chunks)]
    pool = Pool(NUM_WORKER)
    results = pool.map(process_chunk, chunk_data)
    pool.close()
    pool.join()

    # Flatten the list of results
    flat_results = [item for sublist in results for item in sublist]
    save_data(flat_results, args.save_path, jsonl=True)
    key_list = [sample['id'] for sample in flat_results]
    key_list = sorted(key_list)
    pred = {sample['id']: sample for sample in flat_results}
    ordered = [pred[idx] for idx in key_list]
    save_data(ordered, args.save_path, jsonl=True)


def process_chunk(chunk_data):
    args, index, chunk, GPUs = chunk_data
    print(f"Worker:{index}|Sample:{len(chunk)}|GPU:{GPUs}")
    tokenizer, model = get_model(base_model=args.model_path, device=GPUs)

    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        max_length=args.max_len,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        repetition_penalty=1.0,
    )

    result = []

    for sample in tqdm(chunk, disable=False):
        prompt = sample['instruction']
        prompt_batch = [make_chat_prompt(prompt, tokenizer)]
        encoding = tokenizer(prompt_batch, return_tensors="pt", truncation=True, max_length=args.max_len)
        gen_tokens = model.generate(
            input_ids=encoding["input_ids"].to(model.device),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            generation_config=generation_config
        )
        s = gen_tokens[0]
        length = encoding["input_ids"].shape[1]
        output = tokenizer.decode(s[length:], skip_special_tokens=True)
        print(f"==========={sample['id']}===========")
        print(output)
        sample['output'] = output
        sample['generator'] = args.model_name
        result.append(sample)
    return result


def get_num_gpus():
    try:
        n = len(subprocess.check_output(['nvidia-smi', '-L']).decode('utf-8').strip().split('\n'))
    except OSError:
        n = 0
    return n


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='bigcode/starcoder', help="")
    parser.add_argument('--num_gpus', type=int, default=4, help="")
    parser.add_argument('--max_len', type=int, default=4096, help="")
    parser.add_argument('--overwrite', action='store_true', help='')
    parser.add_argument('--gpu_id', type=str, default='4,5,6,7', help="")
    parser.add_argument(
        '--result_root',
        type=str,
        default="result/")
    args = parser.parse_args()
    if args.gpu_id == 'default':
        args.num_gpus = get_num_gpus()
        args.gpu_id = [str(i) for i in range(args.num_gpus)]
    else:
        args.gpu_id = args.gpu_id.split(',')
        args.num_gpus = len(args.gpu_id)
        args.num_workers = args.num_gpus
    args.model_name = args.model_path.split('/')[-1]
    args.result_root = os.path.abspath(args.result_root)
    data_name = 'alpaca_eval'
    args.data_path = os.path.abspath(f"data/eval/{data_name}.json")
    args.save_path = f"{args.result_root}{args.model_name}_{data_name}.jsonl"
    run(args)
