import argparse
import pprint
import os
from tqdm import tqdm
import torch
from multiprocessing import Pool, set_start_method
from utils import load_data, save_data, get_model, get_num_gpus, encoder
import random
import json

set_start_method('spawn', force=True)


def make_prompt(sample):
    # get instructions from sample
    instruction = sample['instruction']
    return instruction


def random_drop_word(text, drop_rate):
    text_split = text.split()
    new_text_split = [t for t in text_split if random.uniform(0, 1) > drop_rate]
    return " ".join(new_text_split)


def perturb_prompt(prompt, n=20, dropout=0.2):
    prompt_list = [prompt]
    for i in range(n):
        prompt_list.append(random_drop_word(prompt, dropout))
    return prompt_list


@torch.no_grad()
def process_chunk(chunk_data):
    args, index, chunk, GPU = chunk_data
    print(f"Worker:{index}|Sample:{len(chunk)}|GPU:{GPU}")
    tokenizer, model = get_model(base_model=args.model_path, device=GPU, args=args)
    message_encoder = encoder(tokenizer, args.template, args.max_len)
    result = []
    for sample in tqdm(chunk, disable=False):
        prompt = make_prompt(sample)
        self_response = sample['evol_info']['uncertainty']['self_response']
        gpt_response = sample['response']
        perturbed = perturb_prompt(prompt, n=10, dropout=0.2)
        # regularize the answer
        self_uncertainty = []
        gpt_uncertainty = []
        for p in perturbed:
            self_message = [
                {'role': 'user', 'content': p},
                {'role': 'assistant', 'content': self_response}
            ]
            gpt_message = [
                {'role': 'user', 'content': p},
                {'role': 'assistant', 'content': gpt_response}
            ]
            self_model_inputs = message_encoder(self_message)
            for key in self_model_inputs:
                self_model_inputs[key] = self_model_inputs[key].to(model.device)
            self_output = model(**self_model_inputs).loss
            # get length of the response, notice when over length, response length will be different
            unc1 = torch.exp(-self_output).cpu()
            self_uncertainty.append(unc1)

            gpt_model_inputs = message_encoder(gpt_message)
            for key in gpt_model_inputs:
                gpt_model_inputs[key] = gpt_model_inputs[key].to(model.device)
            gpt_output = model(**gpt_model_inputs).loss
            # get length of the response, notice when over length, response length will be different
            unc2 = torch.exp(-gpt_output).cpu()
            gpt_uncertainty.append(unc2)

        self_uncertainty = torch.tensor(self_uncertainty)
        self_p_resp = self_uncertainty[0].cpu()
        self_p_unc = torch.mean(abs(self_uncertainty[1:] - self_p_resp)).cpu()

        gpt_uncertainty = torch.tensor(gpt_uncertainty)
        gpt_p_resp = gpt_uncertainty[0].cpu()
        gpt_p_unc = torch.mean(abs(gpt_uncertainty[1:] - gpt_p_resp)).cpu()

        sample['evol_info']['uncertainty']['self'] = [float(self_p_resp), float(self_p_unc)]
        sample['evol_info']['uncertainty']['gpt'] = [float(gpt_p_resp), float(gpt_p_unc)]
        result.append(sample)
        with open(args.cache_path, 'a') as file:
            file.write(json.dumps(sample) + '\n')
    return result


def run(args):
    NUM_GPU = args.num_gpus
    NUM_WORKER = args.num_workers
    print(f"There are {NUM_GPU} GPUs available!")
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # load data
    data = load_data(args.data_path)

    num_samples = len(data)
    print("Number of samples: {}".format(num_samples))

    # load cache
    try:
        cache = load_data(args.cache_path)
        print(f"{len(cache)} samples loaded from {args.cache_path}")
    except:
        with open(args.cache_path, 'w') as f:
            pass
        cache = []
    done = []
    keys = {sample['id']: 0 for sample in data}
    for sample in cache:
        if sample['id'] in keys:
            done.append(sample)
    keys = {sample['id']: 0 for sample in cache}
    remain = [sample for sample in data if sample['id'] not in keys]
    random.shuffle(remain)
    chunks = [remain[i::NUM_WORKER] for i in range(NUM_WORKER)]
    GPU_index = args.gpu_id
    GPUs = GPU_index
        # [GPU_index[i::NUM_WORKER] for i in range(NUM_WORKER)]
    # GPUs = [[str(i // 2)] for i in range(NUM_WORKER)]
    print(f"Seperate Data into {len(chunks)} chunks")
    chunk_data = [(args, index, chunk, GPUs[index]) for index, chunk in enumerate(chunks)]
    pool = Pool(NUM_WORKER)
    results = pool.map(process_chunk, chunk_data)
    pool.close()
    pool.join()
    # Flatten the list of results
    flat_results = [item for sublist in results for item in sublist]
    done += flat_results
    save_data(done, args.save_path)


def update_args(args):
    args.model_name = args.model_path.split('/')[-1]
    args.cache_path = args.cache_path.replace('[]', f"{args.model_name}-{args.use_resp}")
    args.save_path = args.save_path.replace('[]', f"{args.model_name}-{args.use_resp}")
    args.cache_path = os.path.abspath(args.cache_path)
    if args.gpu_id == 'default':
        args.num_gpus = get_num_gpus()
        args.gpu_id = [str(i) for i in range(args.num_gpus)]
    else:
        args.gpu_id = args.gpu_id.split(',')
        args.num_gpus = len(args.gpu_id)
    if args.num_workers == -1:
        args.num_workers = args.num_gpus
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/search/ai/jiuding/LLM_ZOO/llama3/Meta-Llama-3-8B-Instruct',
                        help="")
    parser.add_argument('--data_path', type=str, default='data/amb/cache/d1.json', help="")
    parser.add_argument('--save_path', type=str, default='data/amb/d1_amb_[].json', help="")
    parser.add_argument('--cache_path', type=str, default='cache/amb/d1_amb_[].json', help="")
    parser.add_argument('--template', type=str, default='llama3', help="")
    parser.add_argument('--gpu_id', type=str, default='4,5,6,7', help="")
    parser.add_argument('--num_workers', type=int, default=-1, help="")
    parser.add_argument('--max_len', type=int, default=4096, help="")
    args = parser.parse_args()
    args = update_args(args)
    run(args)
