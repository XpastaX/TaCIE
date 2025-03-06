import tiktoken
from tqdm import tqdm
import requests


def estimate_bill(prompt=None, response=None, model=None, tqdm_disable=False):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    sum_token_prompt = 0
    sum_token_response = 0
    if prompt is not None:
        print('Tokenizing prompts')
        for text in tqdm(prompt, disable=tqdm_disable):
            sum_token_prompt += len(tokenizer.encode(text))
    if response is not None:
        print('Tokenizing responses')
        for text in tqdm(response, disable=tqdm_disable):
            sum_token_response += len(tokenizer.encode(text))

    print(f"prompt_token:{sum_token_prompt}|response_token:{sum_token_response}")
    print(model)
    price_list = {
        'gpt-4o-2024-05-13': [5, 15],
        'gpt-4-1106-preview': [10, 30],
        'gpt-4-0125-preview': [10, 30],
        'gpt-4-turbo-2024-04-09': [10, 30],
        'gpt-4': [30, 60],
        'gpt-4-32k': [60, 120],
        'gpt-3.5-turbo-0125': [0.5, 1.5],
        'gpt-3.5-turbo-1106': [1, 2],
        'gpt-3.5-turbo-0613': [1.5, 2],
        'gpt-3.5-turbo-instruct': [1.5, 2],

    }
    bill = {}
    for mod in price_list:
        bill[mod] = sum_token_prompt / 1000000 * price_list[mod][0] + sum_token_response / 1000000 * price_list[mod][1]
    print("{:^23}: {:^10} {:^10}".format('model', 'USD', 'CYN'))
    if model is None:
        for mod in bill:
            print(
                "{:<23}: {:<10} {:<10}".format(mod, str(round(bill[mod], 2)), round(convert_usd_to_rmb(bill[mod]), 2)))
        return bill
    else:
        print("{:<23}: {:<10} {:<10}".format(model, str(round(bill[model], 2)),
                                             round(convert_usd_to_rmb(bill[model]), 2)))
        return bill[model]


def convert_usd_to_rmb(amount):
    # API endpoint for currency conversion
    api_url = "https://api.exchangerate-api.com/v4/latest/USD"

    try:
        # Sending a request to the API
        response = requests.get(api_url)
        data = response.json()

        # Getting the exchange rate for USD to RMB (CNY)
        exchange_rate = data['rates']['CNY']

        # Calculating the converted amount
        converted_amount = amount * exchange_rate

        return converted_amount
    except Exception as e:
        return -1


def check_model(model_name):
    import openai
    # check model
    model_list = openai.Model.list()['data']
    all_models = [item['id'] for item in model_list]
    if type(model_name) is str:
        if model_name not in all_models:
            print("model not available")
            raise NotImplementedError
        return
    else:
        for name in model_name:
            if name not in all_models:
                print("model not available")
                raise NotImplementedError
