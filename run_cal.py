import argparse
import os
import pprint
import random
import subprocess
import torch
from transformers import AutoTokenizer
import json
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, set_start_method
import config as cfg_ori
from utils import check_file, set_seed, check_dir, load_data, save_data, get_model, encoder
from time import sleep

set_start_method('spawn', force=True)


def get_file_name(path_list):
    return [os.path.basename(path) for path in path_list]


def check_run(cfg):
    root_to_cal = cfg.comm_to_cal
    root_gen_resp = cfg.comm_gen_resp
    root_cal_done = cfg.comm_cal_done
    to_cal = glob(root_to_cal + '*.json')
    generated = glob(root_gen_resp + "*.json")
    done = glob(root_cal_done + "*.json")

    name_to_cal = get_file_name(to_cal)
    name_generated = get_file_name(generated)
    name_done = get_file_name(done)

    for index, name in enumerate(name_to_cal):
        if name not in name_generated:
            return True
    for index, name in enumerate(name_generated):
        if name not in name_done:
            print(f'Calculating Uncertainty for {name}')
            return True
    return False


def run(cfg):
    root_to_cal = cfg.comm_to_cal
    root_gen_resp = cfg.comm_gen_resp
    root_cal_done = cfg.comm_cal_done
    print("checking files...")
    while not check_file(cfg.final_data_path):

        to_cal = glob(root_to_cal + '*.json')
        generated = glob(root_gen_resp + "*.json")
        done = glob(root_cal_done + "*.json")

        name_to_cal = get_file_name(to_cal)
        name_generated = get_file_name(generated)
        name_done = get_file_name(done)

        # n = 64 * 10 ** 8
        # num_gpu = torch.cuda.device_count()
        # if 'CUDA_VISIBLE_DEVICES' in os.environ:
        #     del os.environ['CUDA_VISIBLE_DEVICES']
        # while not check_run(cfg):
        #     for i in range(num_gpu):
        #         device = torch.device(f"cuda:{i}")
        #         shape = (n,)
        #         x = torch.randn(shape, dtype=torch.float32, device=device)
        # x = None
        # torch.cuda.empty_cache()
        # print('Find new file!')

        for index, name in enumerate(name_to_cal):
            if name not in name_generated:
                print(f'Generating Response for {name}')
                sleep(10)
                gen(cfg, args, name)
        for index, name in enumerate(name_generated):
            if name not in name_done:
                print(f'Calculating Uncertainty for {name}')
                sleep(10)
                cal(cfg, args, name)


def gen(cfg, args, name):
    args.data_path = os.path.abspath(cfg.comm_to_cal + f"{name}")
    args.save_path = os.path.abspath(cfg.comm_gen_resp + f"{name}")
    args.cache_path = os.path.abspath(cfg.cal_cache + f"gen_cache_{name}")
    args.gpu_id = cfg.gpu_id
    args.max_len = cfg.max_len
    args.model_path = cfg.model_path
    args.num_workers = 4
    args.num_gpus = 4
    run_gen(args)


def cal(cfg, args, name):
    args.data_path = os.path.abspath(cfg.comm_gen_resp + f"{name}")
    args.save_path = os.path.abspath(cfg.comm_cal_done + f"{name}")
    args.cache_path = os.path.abspath(cfg.cal_cache + f"cal_cache_{name}")
    args.gpu_id = cfg.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpu_id)
    args.max_len = cfg.max_len
    args.model_path = cfg.model_path
    args.template = 'llama3'
    args.num_workers = 4
    args.num_gpus = 4
    run_cal(args)


def run_gen(args):
    print('================================')
    NUM_GPU = args.num_gpus
    print(f"There are {NUM_GPU} GPUs available!")
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # load data
    success = False
    while not success:
        try:
            data = load_data(args.data_path)
            if len(data)>0:
                success=True
        except:
            print(f"Load data failed: {args.data_path}")
            sleep(10)
    # load cache
    result = []
    # collect all sample keys
    num_samples = len(data)
    print("Number of Remaining Samples: {}".format(num_samples))

    # shuffle remain to ensure a similar task distribution on each GPU
    random.shuffle(data)
    # load tokenizer and apply template
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": sample['instruction']}], tokenize=False,
                                             add_generation_prompt=True) for sample in data]
    print(f"Total number of instructions: {len(prompts)}")

    samples_per_gpu = len(prompts) // NUM_GPU
    processes = []
    cache_path = args.cache_path
    cache_dir = cache_path[:-len(".json")] + '/'
    print(f"Cache Dir: {cache_dir}")
    check_dir(cache_dir)

    for i, gpu_id in enumerate(args.gpu_id):
        start_idx = i * samples_per_gpu
        end_idx = (i + 1) * samples_per_gpu if i != NUM_GPU - 1 else len(prompts)
        remain_chunk = data[start_idx:end_idx]
        prompts_chunk = prompts[start_idx:end_idx]

        para_path = cache_dir + f'{gpu_id}.torch'
        sub_cache_path = cache_dir + f"{gpu_id}.json"
        with open(sub_cache_path, 'w'):
            pass
        torch.save((remain_chunk, prompts_chunk, args), para_path)

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        process = subprocess.Popen(['python', 'gen_vllm.py',
                                    '--gpu_id', str(gpu_id),
                                    '--para_path', para_path,
                                    '--cache_path', sub_cache_path], env=env)
        processes.append(process)
    # Wait for all processes to complete
    for p in processes:
        p.wait()
    cache = []
    for gpu_id in args.gpu_id:
        tmp = load_data(cache_dir + f"{gpu_id}.json")
        cache += tmp
    print(f"Saving generation result to {args.save_path}")

    save_data(cache, args.save_path)


def run_cal(args):
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']
    NUM_GPU = args.num_gpus
    NUM_WORKER = args.num_workers
    print(f"There are {NUM_GPU} GPUs available!")
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # load data
    success = False
    while not success:
        try:
            data = load_data(args.data_path)
            if len(data) > 0:
                success = True
        except:
            print(f"Load data failed: {args.data_path}")
            sleep(10)

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
    GPUs = args.gpu_id
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


if __name__ == '__main__':
    set_seed(123)
    parser = argparse.ArgumentParser(description="Calculate uncertainty")
    parser.add_argument('--model_path', type=str, default='',
                        help="")
    parser.add_argument('--gpu_id', type=str, default='', help="")
    parser.add_argument('--max_len', type=int, default=-1, help="")
    args = parser.parse_args()
    check_dir(cfg_ori.cal_cache)
    if args.model_path != '':
        cfg_ori.model_path = args.model_path
    if args.gpu_id != '':
        cfg_ori.gpu_id = args.gpu_id
    if args.max_len != -1:
        cfg_ori.max_len = args.max_len
    run(cfg_ori)
