# coding: utf-8

import time
import torch

# 获取可用的GPU数量
num_gpu = torch.cuda.device_count()

n = 16 * 10 ** 8
seg = 0.1

# 遍历所有可用的GPU
while 1:
    for i in range(num_gpu):
        device = torch.device(f"cuda:{i}")
        shape = (n, )
        x = torch.randn(shape, dtype=torch.float32, device=device)
        # print(torch.sum(x))
        # time.sleep(seg)
# "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
