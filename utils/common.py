import torch
import random
import os
import numpy as np
import json


def check_file(path):
    return os.path.isfile(path)


def check_dir(path, creat=True, force=False):
    path = os.path.split(path)[0]
    if not os.path.exists(path):
        if creat:
            os.makedirs(path)
            print('Folder %s has been created.' % path)
            return True
        else:
            return False
    else:
        if force:
            os.makedirs(path)
            print('Force to create %s.' % path)
        return True


def print_args(args):
    for _arg in args._get_kwargs():
        print(f"{_arg[0]}:{_arg[1]}")


def print_config(cfg, ban):
    # List all attributes of the module
    all_variables = dir(cfg)

    # Filter out built-in attributes
    variables = [var for var in all_variables if not var.startswith("__")]

    # Print the name and value of each variable
    for var in variables:
        isPrint = True
        for pattern in ban:
            if pattern in var:
                isPrint = False
        if isPrint:
            print(f"{var}: {getattr(cfg, var)}")


def print_dict(obj: dict):
    for key in obj:
        print(f"{key}:{obj[key]}")


def set_seed(seed):
    """
    :param seed:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def write_log(txt, path, prt=True, creat=False):
    if prt:
        print(txt)
    if txt[-1:] != '\n':
        txt += '\n'
    if not creat:
        with open(path, 'a') as file:
            file.writelines(txt)
    else:
        with open(path, 'w') as file:
            file.writelines(txt)


def print_class(obj):
    tmp = {name: value for name, value in obj.__dict__.items() if '__' not in name}
    txt = ''
    for key in tmp:
        txt += f'{key}:{tmp[key]}\n'
    return txt


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


class logger(object):
    def __init__(self, path, create=True):
        self.path = path
        if create:
            with open(self.path, 'w', encoding='UTF-8') as f:
                pass

    def log(self, txt, prt=True):
        if prt:
            print(txt)
        self.write(txt)

    def write(self, txt):
        with open(self.path, 'a', encoding='UTF-8') as f:
            f.write(txt+'\n')

    def print(self):
        with open(self.path, 'r') as f:
            txt = f.read()
        print(txt)


def sort_list_and_return_indices(input_list):
    # Pair each element with its index
    indexed_list = list(enumerate(input_list))

    # Sort the indexed list by the values
    sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1])

    # Extract the sorted indices
    sorted_indices = [index for index, value in sorted_indexed_list]

    return sorted_indices