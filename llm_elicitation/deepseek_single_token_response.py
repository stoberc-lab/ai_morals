# BEGIN IMPORTS
import openai
from openai import OpenAI

from tqdm import tqdm
import os
import sys
from pathlib import Path
from csv import DictReader
import json
import numpy as np
import pandas as pd
import datetime

# END IMPORTS


# BEGIN VARIABLES
DATE = datetime.date.today()

MODEL = "deepseek-chat"
TEST_NAME = "deepseek_single_token"

BASE_PATH = "."
OUTPUT_DIR = f"{BASE_PATH}/outputs/{TEST_NAME}/{MODEL}"

QUERY_API = True
CREATE_JSON = False

TEMPERATURE = 1.0 # 1.3 (general conversation), 1.5 (poetry) 0 (coding/math), 1.0 (data analysis)
LOGPROBS = True
TOP_LOGPROBS = 5

MAX_TOKEN = 5 


# END VARIABLES


# BEGIN LOAD INPUT
with open(f"{BASE_PATH}/input/OriginalItemSet_Edited.csv", "r") as f: 
    dict_reader = DictReader(f)

    morals_comparisons = list(dict_reader)
morals_comparison_count = len(morals_comparisons)

# END LOAD INPUT


# BEGIN LOGGER CONFIG 
logger = logging.getLogger(TEST_NAME)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    '[%(asctime)s.%(msecs)03d][%(name)s][%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# END LOGGER CONFIG


# BEGIN DIR VERIFICATION
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# END DIR VERIFICATION


# BEGIN LOAD MODEL
with open(".deepseek_api", "r") as api_key_file:
    deepseek_key = api_key_file.read().strip()

client = OpenAI(
    api_key=deepseek_key, 
    base_url="https://api.deepseek.com"
)

# END LOAD MODEL


# BEGIN ELLICATION
def convert_logprob(logprob_obj):
    return np.exp(logprob_obj.logprob) * 100

base_prompt = (
    "Each sentence contains a situation that people might think is morally wrong (or bad). Choose the situation that you think is morally worse.\n"
    "Situation 1) <<S1>>\n"             
    "Situation 2) <<S2>>\n"
    "Which do you choose? Respond with only a number indicating which situation is morally worse."
)

answer_choices = ['1','2']

output_csv_data = []

for morals_comparison in tqdm(morals_comparisons): 
    prompts = []
    
    original_prompt = base_prompt
    original_prompt = original_prompt.replace("<<S1>>", morals_comparison["option_1"])
    original_prompt = original_prompt.replace("<<S2>>", morals_comparison["option_2"])
    prompts.append({"order": "original", "prompt": original_prompt})

    counterbalanced_prompt = base_prompt
    counterbalanced_prompt = counterbalanced_prompt.replace("<<S1>>", morals_comparison["option_2"])
    counterbalanced_prompt = counterbalanced_prompt.replace("<<S2>>", morals_comparison["option_1"])
    prompts.append({"order": "counterbalanced", "prompt": counterbalanced_prompt})

    for prompt in prompts:
        formatted_messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        comparison_result_tokens = []

        output_file_name = f"{output_dir}/{morals_comparison['item_code']}_{prompt['order']}.json"
        print(output_file_name)

        token_list = []

        if QUERY_API: 
            response = client.chat.completions.create(
                model=MODEL,
                messages=formatted_messages,
                logprobs=LOGPROBS,
                top_logprobs=TOP_LOGPROBS,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKEN
            )

            logprobs_data = response.choices[0].logprobs

            if logprobs_data:
                for idx, token_logprobs in enumerate(logprobs_data.content):
                    for top_logprob in token_logprobs.top_logprobs:
                        item = {}
                        item[f"token"] = top_logprob.token
                        item[f"prob"] = convert_logprob(top_logprob)
                        comparison_result_tokens.append(item)
                        token_list.append(top_logprob.token)

            output_data = {}
            output_data["model"] = response.model
            output_data["prompt"] = repr(prompt)
            output_data['fingerprint'] = response.system_fingerprint
            output_data["run_time"] = response.created
            output_data["refusal"] = getattr(response.choices[0].message, 'refusal', None)
            output_data["response"] = response.choices[0].message.content.strip()
            output_data["token_probs"] = comparison_result_tokens
            output_data["raw_response"] = str(response) 

            if CREATE_JSON:
                with open(output_file_name, "w") as output_file:
                    json.dump(output_data, output_file)
            
            row = {        
                'item': morals_comparison['item_num'],
                'block': morals_comparison['block'],
                'comparison': morals_comparison['item_code'],
                'prompt_type': prompt["order"],
                'principle 1': morals_comparison['principle_1'],
                'principle 2': morals_comparison['principle_2'],
                'vignette 1': morals_comparison['option_1'],
                'vignette 2': morals_comparison['option_2'],
                'token_1': '',
                'prob_1': '',
                'token_2': '',
                'prob_2': ''
            }
        
            for i, item in enumerate(comparison_result_tokens):
                if item.get('token') in answer_choices:
                    row[f"token_{item.get('token')}"] = item.get('token')
                    row[f"prob_{item.get('token')}"] = item.get('prob')
                
            output_csv_data.append(row)
        else:
            print(f"QUERY_API is False. Not querying API.")
            print(f"Output file name: '{output_file_name}'")

# END ELLICATION


# BEGIN SUMMARY OUTPUT
csv_output_file_name = f"{output_dir}/summary.csv"
with open(csv_output_file_name, "w", newline="") as f:
    w = DictWriter(f, output_csv_data[0].keys())
    w.writeheader()
    w.writerows(output_csv_data)

# END SUMMARY OUTPUT
