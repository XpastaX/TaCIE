import json
import random
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from multiprocessing import Process
from utils import load_data, save_data  # Replace with actual imports
from vllm import LLM, SamplingParams
import torch


def inference_on_gpu(remain, prompts, args):
    llm = LLM(model=args.model_path, tensor_parallel_size=1, trust_remote_code=True, enforce_eager=False,
              tokenizer_mode='auto', dtype='float16')

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


def main():
    parser = argparse.ArgumentParser(description="Run inference on GPU.")
    parser.add_argument('--para_path', type=str, required=True, help="Path to the remaining samples")
    parser.add_argument('--cache_path', type=str, required=True, help="Path to the remaining samples")
    parser.add_argument('--gpu_id', type=str, required=True, help="")

    args = parser.parse_args()
    gpu_id = args.gpu_id
    cache_path = args.cache_path
    # Load remain and prompts data
    remain, prompts, args = torch.load(args.para_path)
    args.gpu_id = gpu_id
    args.cache_path = cache_path
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"worker GPU: {args.gpu_id} | cache: {args.cache_path}")
    inference_on_gpu(remain, prompts, args)


if __name__ == "__main__":
    main()
