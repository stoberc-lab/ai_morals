# BEGIN IMPORTS
import openai


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

MODEL = "gpt-5.2"
TEST_NAME = "openai_single_token"

BASE_PATH = "."
OUTPUT_DIR = f"{BASE_PATH}/outputs/{TEST_NAME}/{MODEL}"

QUERY_API = True
CREATE_JSON = False
TEMPERATURE = 0.0
LOGPROBS = True
TOP_LOGPROBS = 5

MAX_TOKENS = TOP_LOGPROBS + 1 # N LogProbs tokens + 1 response token

# END VARIABLES


# BEGIN LOAD INPUT
with open("OriginalItemSet_Edited.csv", 'r') as f:
    dict_reader = DictReader(f)
    
    morals_comparisons = list(dict_reader)

# END LOAD INPUT


# BEGIN DIR VERIFICATION
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# END DIR VERIFICATION


# BEGIN LOAD MODEL
with open(".openai_api", "r") as api_key_file:
    openai.api_key = api_key_file.read()

# END LOAD MODEL


# BEGIN ELLICATION
def convert_logprob(logprob):
    return np.exp(logprob.logprob)*100

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
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        # print(formatted_messages)

        comparison_result_tokens = []

        output_file_name = f"{output_dir}/{morals_comparison['item_code']}.json"
        print(output_file_name)

        if QUERY_API:
            response = openai.chat.completions.create(
                model=MODEL,
                messages=formatted_messages,
                logprobs=LOGPROBS,
                top_logprobs=TOP_LOGPROBS,
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_TOKENS
            )

            logprobs_data = response.choices[0].logprobs

            # Display the top 5 tokens and their probabilities for the first token
            for idx, token_logprobs in enumerate(logprobs_data.content):
                for top_logprob in token_logprobs.top_logprobs:
                    item = {}
                    item[f"token"] = top_logprob.token
                    item[f"prob"] = convert_logprob(top_logprob)
                    comparison_result_tokens.append(item)

            output_data = {}
            # print(response)
            output_data["model"] = response.model
            output_data["prompt"] = repr(prompt)
            output_data["fingerprint"] = response.system_fingerprint
            output_data["run_time"] = response.created
            output_data["refusal"] = response.choices[0].message.refusal
            output_data["response"] = response.choices[0].message.content.strip()
            output_data["token_probs"] = comparison_result_tokens
            output_data["raw_response"] = str(response)

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

            answer_choices = ['1','2']

            for i, item in enumerate(comparison_result_tokens):
                if item.get('token') in answer_choices:
                    row[f"token_{item.get('token')}"] = item.get('token')
                    row[f"prob_{item.get('token')}"] = item.get('prob')
            
            output_csv_data.append(row)
        else:
            print(f"QUERY_API is False. Not querying API.")
            print(f"Output file name: '{output_file_name}'")

# END ELLICATION


# BEGIN SUMMARY DATA
df = pd.DataFrame(output_csv_data)
excel_file_name = f"{output_dir}/{DATE}_comparisons.csv" #_cb.csv"
df.to_csv(excel_file_name, index=False, encoding='UTF-8')

# END SUMMARY DATA
