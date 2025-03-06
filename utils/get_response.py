import time
import json
import openai
from utils.data import load_data
from multiprocessing import Pool
from tqdm import tqdm

# Initialize OpenAI client with API key
openai.api_key = "your_openai_api_key_here"


def replace(sample, prefix, replace_dict):
    input_text = prefix
    for prefix_key, replace_key in replace_dict.items():
        input_text = input_text.replace(prefix_key, sample[replace_key])
    return input_text


def crawl_with_instruct(data, args):
    # Load cache file
    prefix = args.prefix
    replace_dict = args.replace_dict
    try:
        cache = load_data(args.cache_file_path)
        print(f"{len(cache)} samples loaded from {args.cache_file_path}")
    except FileNotFoundError:
        print(f"Load cache failed, creating new cache at {args.cache_file_path}")
        with open(args.cache_file_path, 'w') as file:
            pass
        cache = []

    cache_key = {sample['id']: 1 for sample in cache}
    data_key = {sample['id']: 1 for sample in data}
    remain = [sample for sample in data if sample['id'] not in cache_key]

    args_list = [(sample, replace(sample, prefix, replace_dict), args.model_name,
                  args.cache_file_path, args.response_store_key) for sample in remain]

    results = [sample for sample in cache if sample['id'] in data_key]

    with Pool(args.num_workers) as pool:
        tasks = [pool.apply_async(crawl_with_instruct_unit, args=(_arg,)) for _arg in args_list]
        for task in tqdm(tasks, total=len(tasks)):
            result = task.get()
            if result is not None:
                results.append(result)
    return results


def crawl_with_instruct_unit(args):
    sample, prompt, model_name, tmp_file, response_store_key = args
    if response_store_key in sample and sample[response_store_key] is not None:
        return sample

    resp = get_response(prompt, model_name)
    if resp is not None:
        sample[response_store_key] = resp
        with open(tmp_file, 'a') as file:
            file.write(json.dumps(sample) + '\n')
    return sample


def get_response(prompt, model_name, system=None, message=None):
    if message is None:
        message = [{'role': 'system', 'content': system}] if system else []
        message.append({'role': 'user', 'content': prompt})

    count = 4
    while count > 0:
        try:
            resp = get_response_openai(message, model_name)
            return resp
        except Exception as e:
            print(f"Error: {e}. Retrying... ({count} attempts left)")
            time.sleep(1)
            count -= 1
    return None


def get_response_openai(messages, model_name, max_tokens=4096, temperature=0.7):
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        raise
