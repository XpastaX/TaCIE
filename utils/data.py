import json
import tiktoken
import random

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


def read_file(path):
    with open(path, 'r') as f:
        txt = f.read()
    return txt


def filter_with_max_token(data):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    new_data = []
    for sample in data:
        if len(tokenizer.encode(sample['instruction'])) < 2048:
            new_data.append(sample)
    return new_data


def flatten_data(data: dict):
    new_data = []
    maps = {}
    for key in data:
        colle = data[key]
        for sample in colle:
            new_data.append(sample)
            maps[sample['id']] = key
    return new_data, maps


def reconstruct_data(data: dict, result: list, maps: dict):
    for sample in result:
        if 'response' not in sample:
            print("detect_failed_response")
            continue
        key = maps[sample['id']]
        for i, sample2 in enumerate(data[key]):
            if sample2['id'] == sample['id']:
                data[key][i] = sample
    return data


# Function to sample unique pairs of data from the list
def sample_unique_pairs_with_seed(data, n_pairs=1, seed=None):
    if seed is not None:
        random.seed(seed)

    sampled_pairs = set()
    index = list(range(len(data)))
    while len(sampled_pairs) < n_pairs:
        pair = tuple(sorted(random.sample(index, 2)))
        sample1 = data[pair[0]]
        sample2 = data[pair[1]]
        if sample1['id'][:sample1['id'].index('|')] != sample2['id'][:sample2['id'].index('|')]:
            sampled_pairs.add(pair)

    colle = []
    for (i,j) in sampled_pairs:
        colle.append([data[i], data[j]])
    return colle


# Function to sample 2000 unique pairs of elements from every pair of sub-dicts
def sample_pairs_from_subdicts(data, n_pairs=2000, seed=None):
    if seed is not None:
        random.seed(seed)

    sub_dict_keys = list(data.keys())
    sampled_pairs = {}

    for i in range(len(sub_dict_keys)):
        for j in range(i + 1, len(sub_dict_keys)):
            key1 = sub_dict_keys[i]
            key2 = sub_dict_keys[j]
            sampled_pairs[(key1, key2)] = []

            seen_pairs = set()
            while len(sampled_pairs[(key1, key2)]) < n_pairs:
                element1 = random.choice(list(data[key1].keys()))
                element2 = random.choice(list(data[key2].keys()))
                pair = (random.choice(data[key1][element1]), random.choice(data[key2][element2]))
                if (element1, element2) not in seen_pairs:
                    seen_pairs.add((element1, element2))
                    sampled_pairs[(key1, key2)].append(pair)

    return sampled_pairs

def split_data(data):
    # split data into tasks
    pass