from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import subprocess

B_INST, E_INST = "[INST] ", " [/INST] "
IGNORE_INDEX = -100


def apply_LLM_template(txt, template, tokenizer=None):
    # get prompt for input instruction
    if template == 'llama':
        return B_INST + txt + E_INST
    elif template == 'mistral':
        return "[INST] " + txt + " [/INST]"
    elif template in ['qwen', 'llama3']:
        messages = [
            {"role": "user", "content": txt}
        ]
        texts = tokenizer.apply_chat_template(
            messages,
            # chat_template=TEMPLATE,
            tokenize=False,
            add_generation_prompt=True,
            padding=False,
            max_length=4096,
            truncation=True)
        return texts
    else:
        return txt


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


def encode_prompt(tokenizer, prompt):
    input_ids = tokenizer.encode(
        text=prompt,
        add_special_tokens=False
    )
    if input_ids[0] != tokenizer.bos_token_id:
        input_ids = [tokenizer.bos_token_id] + input_ids
    model_inputs = torch.tensor([input_ids])
    return model_inputs


def get_num_gpus():
    try:
        n = len(
            subprocess.check_output(['nvidia-smi', '-L']).decode('utf-8').strip().split('\n'))
    except OSError:
        n = 0
    return n


class encoder(object):
    def __init__(self, tokenizer, template_name, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.template_name = template_name

    def encode_msg(self, messages, batch=False):
        model_inputs = {}
        if messages[0]['role'] != "user":
            messages = messages[1:]

        texts_input = self.tokenizer.apply_chat_template(
            messages[:1],
            tokenize=True,
            add_generation_prompt=True,
            padding=False,
            max_length=self.max_seq_length,
            truncation=False)

        texts_all = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            padding=True if not batch else 'max_length',
            max_length=self.max_seq_length,
            truncation=True)
        input_ids = torch.LongTensor(texts_all).cuda()
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX
        labels[:len(texts_input)] = IGNORE_INDEX
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        model_inputs["input_ids"] = input_ids.unsqueeze(0)
        model_inputs["attention_mask"] = attention_mask.unsqueeze(0)
        model_inputs["labels"] = labels.unsqueeze(0)
        return model_inputs

    # def __call__(self, message, template_name=None):
    #     model_inputs = self.encode_msg_qwen(message)
    #     return model_inputs

    def __call__(self, messages, batch=False):
        if batch:
            inputs = [self.encode_msg(message, batch) for message in messages]
            batch_inputs = {}
            batch_inputs["input_ids"] = torch.cat([model_inputs['input_ids'] for model_inputs in inputs])
            batch_inputs["attention_mask"] = torch.cat([model_inputs['attention_mask'] for model_inputs in inputs])
            batch_inputs["labels"] = torch.cat([model_inputs['labels'] for model_inputs in inputs])
            # cut to max_batch_length
            # check = -100 * torch.ones_like(batch_inputs["labels"][:, 0])
            # for i in range(-1, batch_inputs["labels"].shape[-1], -1):
            #     if batch_inputs["labels"][:, i] != check:
            #         batch_inputs["labels"] = batch_inputs["labels"][:, :i + 1]
            #         batch_inputs["attention_mask"] = batch_inputs["attention_mask"][:, :i + 1]
            #         batch_inputs["input_ids"] = batch_inputs["input_ids"][:, :i + 1]
            #         print(i)
            #         break
            return batch_inputs
        else:
            return self.encode_msg(messages)
