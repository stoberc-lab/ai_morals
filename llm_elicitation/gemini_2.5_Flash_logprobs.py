# BEGIN IMPORTS
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig

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

MODEL = "gemini-2.5-flash" ## most up-to-date model that allows extraction of logprobs and controlling for reasoning
TEST_NAME = "gemini_single_token"

BASE_PATH = "."
OUTPUT_DIR = f"{BASE_PATH}/outputs/{TEST_NAME}/{MODEL}"

QUERY_API = True
CREATE_JSON = False
TEMPERATURE = 1.0
LOGPROBS = True
TOP_LOGPROBS = 20

SEED = 1 #sd

# END VARIABLES


# BEGIN LOAD INPUT
with open("OriginalItemSet_Edited.csv", 'r', encoding='latin1') as f:
    dict_reader = DictReader(f)

    morals_comparisons = list(dict_reader)

# END LOAD INPUT


# BEGIN DIR VERIFICATION
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# END DIR VERIFICATION

# BEGIN LOAD MODEL
client = genai.Client(vertexai=True, project="ai-morals", location="us-central1")

# END LOAD MODEL


# BEGIN ELLICATION
def convert_logprob(logprob):
    return np.exp(logprob) * 100

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
        comparison_result_tokens = []

        output_file_name = f"{output_dir}/{morals_comparison['item_code']}.json"
        print(output_file_name)

        if QUERY_API:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=GenerateContentConfig(
                    response_logprobs=LOGPROBS,
                    logprobs=TOP_LOGPROBS,
                    temperature=TEMPERATURE,
                    seed=SEED,
                    candidate_count=1,
                    tools=[],
                    response_modalities=["TEXT"],
                    thinking_config=ThinkingConfig(thinking_budget=0),
                    safety_settings=[
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]
                )
            )

            # Display the top tokens and their probabilities for the first token
            logprobs_data = response.candidates[0].logprobs_result
            
            for idx, token_logprobs in enumerate(logprobs_data.top_candidates):
                for candidate in token_logprobs.candidates:
                    item = {}
                    item[f"token"] = candidate.token
                    item[f"prob"] = convert_logprob(candidate.log_probability)
                    comparison_result_tokens.append(item)

            output_data = {}
            output_data["model"] = response.model_version #closest thing to system_fingerprint
            output_data["prompt"] = repr(prompt)
            output_data["run_time"] = response.create_time
            output_data["finish"] = response.candidates[0].finish_reason
            output_data["response"] = response.candidates[0].content.parts[0].text
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
df = pd.DataFrame(output_csv_data)
excel_file_name = f"{output_dir}/{DATE}_comparisons.csv" #_cb.csv"
df.to_csv(excel_file_name, index=False, encoding='UTF-8')

# END SUMMARY DATA
