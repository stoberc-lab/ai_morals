# BEGIN IMPORTS
from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch

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

MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
TEST_NAME = "llama4_maverick_logprobs"

BASE_PATH = "."
OUTPUT_DIR = f"{BASE_PATH}/outputs/{TEST_NAME}/{MODEL}"

QUERY_API = True
CREATE_JSON = False
TEMPERATURE = 1.0
LOGPROBS = True
TOP_LOGPROBS = 5

MAX_TOKENS = TOP_LOGPROBS + 1 # N LogProbs tokens + 1 response token

# set if you need to store model files from HuggingFace in a specific path, None for default
HF_CACHE_DIR = None

# END VARIABLES


# BEGIN LOAD INPUT
with open("input/OriginalItemSet_Edited.csv", 'r') as f:
    dict_reader = DictReader(f)
    
    morals_comparisons = list(dict_reader)
morals_comparisons_count = len(morals_comparisons)

# END LOAD INPUT


# BEGIN DIR VERIFICATION
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# END DIR VERIFICATION


# BEGIN LOAD MODEL
access_token = ""
access_token_file_paths = [Path(".hf_access_token")]

for access_token_file_path in access_token_file_paths:
    if access_token_file_path.is_file():
        with open(access_token_file_path) as access_token_file:
            access_token = access_token_file.read()
        if access_token == "" or not access_token.startswith("hf_"):
            access_token = ""

hf_access_token = access_token.rstrip()

processor = AutoProcessor.from_pretrained(
    MODEL,
    token=hf_access_token,
    cache_dir=HF_CACHE_DIR,
)

model = Llama4ForConditionalGeneration.from_pretrained(
    MODEL,
    attn_implementation="sdpa",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=hf_access_token,
    cache_dir=HF_CACHE_DIR,
)

# END LOAD MODEL


# BEGIN ELLICATION
def _tensorize(model, processor, formatted_prompt):
    inputs = {}

    continue_final_message = True

    formatted_chat = processor.apply_chat_template(
        formatted_prompt,
        tokenize=True,
        continue_final_message=continue_final_message,
        return_tensors="pt",
        return_attention_mask=True,
    )

    formatted_chat_text = processor.decode(formatted_chat)
    print(formatted_chat_text)
    if (
        formatted_prompt[-1]["content"][0]["text"].endswith("\n\n")
        and not formatted_chat_text[0].endswith("\n\n")
    ):
        formatted_chat = torch.cat(
            (formatted_chat, torch.tensor([[368]])),
            dim=-1,
        )


    print(processor.decode(formatted_chat))

    inputs["input_ids"] = formatted_chat

    inputs["input_ids"] = inputs["input_ids"].to(model.device)
    inputs["attention_mask"] = torch.Tensor(
        [[1.0] * len(inputs["input_ids"][0])]
    ).to(model.device)

    return inputs

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
                        "text": prompt["prompt"]
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "\n\n"
                    }
                ]
            }
        ]

        comparison_result_tokens = []

        output_file_name = f"{OUTPUT_DIR}/{morals_comparison['item_code']}_{prompt['order']}.json"
        print(f"Prompting '{output_file_name}'")
        
        if QUERY_API:
            prompt_tensor = _tensorize(model, processor, formatted_messages)

            with torch.no_grad():
                response = model(**prompt_tensor)

            last_token_logits = response.logits[0, -1, :]

            scores = last_token_logits / TEMPERATURE

            probs = torch.softmax(scores, dim=0, dtype=torch.float64)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)

            continue_zip = True
            start_index = 0
            end_index = 0
            zip_size = 10
            token_probs = {}
            top_k = 5
            min_p = 0

            token_list = []

            while continue_zip and start_index < len(sorted_probs):
                end_index = min(len(sorted_probs), start_index + zip_size)
                if top_k:
                    end_index = min(top_k, end_index)

                sliced_probs = sorted_probs[start_index:end_index]
                sliced_indices = sorted_indices[start_index:end_index]

                start_index += zip_size

                for prob, token in zip(sliced_probs, sliced_indices):
                    if prob.item() < min_p:
                        continue_zip = False
                        break

                    token_probs[token.item()] = (
                        processor.decode([token.item()]),
                        prob.item(),
                    )
                    token_list.append(processor.decode([token.item()]))

            print("Token_probs: ")
            print(token_probs)
            
            # Display the top 5 tokens and their probabilities for the first token
            for idx, token_logprobs in token_probs.items():
                item = {}
                item[f"token"] = token_logprobs[0]
                item[f"prob"] = token_logprobs[1]
                comparison_result_tokens.append(item)

            output_data = {}

            output_data["model"] = MODEL
            output_data["prompt"] = repr(prompt)
            output_data["run_time"] = DATE
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
    break

# END ELLICATION

# BEGIN SUMMARY OUTPUT
df = pd.DataFrame(output_csv_data)
csv_file_name = f"{OUTPUT_DIR}/{DATE}_comparisons.csv" #_cb.csv"
df.to_csv(csv_file_name, index=False, encoding='UTF-8')

# END SUMMARY OUTPUT
