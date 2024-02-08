import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, GPT2Config, get_linear_schedule_with_warmup, AutoTokenizer, TextDataset
from datasets import load_dataset, Dataset
import os
import pandas as pd
from tqdm import tqdm
import wandb
import psutil
from datetime import datetime
import pytz
import json
import sys
import torch.nn.functional as func
from typing import List
import multiprocessing as mp
import math
# use dataloader for batching
from torch.utils.data import DataLoader


#### variables
wandb_run_name = "training_loop"
wandb_project = "gpt-2-finetuning"

uni_delim = "\0" # a unique delimiter to save and load filtered text from dataset
# Here we use "\0" (null character) as it's unlikely to be in the text

####
'''
#### wandb setup

wandb.login()
# initialize WandB
wandb.init(project=wandb_project, name= wandb_run_name)

# using wanb to log ram usage
def log_ram_usage():
    # Get memory usage in percentage
    memory_usage_percent = psutil.virtual_memory().percent
    # Log the memory usage
    wandb.log({"RAM Usage (%)": memory_usage_percent})

#### finished wandb setup
'''

dataset_name = 'wikitext'
dataset_config = 'wikitext-103-v1'
dataset_path_head = os.getcwd()
home_dir = os.getcwd()
dataset_filename = '/wikitext-103-raw-v1'


#### creating input_id and attention_masks tensors 

# loading saved cleaned tokens for test, train, validation sets

# loading raw text and filtering
def load_or_fetch_wikitext(dataset_name, dataset_config, dataset_path):
  # Check if the dataset is already saved locally
  if os.path.exists(dataset_path): # if it is
    print(f"Loading dataset from {dataset_path}")
    # Load the dataset from the file
    dataset = load_dataset(dataset_name, dataset_config)
    return dataset
  else:
    print(f"Fetching dataset from Hugging Face and caching it locally")
    # Load the dataset from Hugging Face and cache it locally
    dataset = load_dataset(dataset_name, dataset_config, cache_dir=dataset_path)
    print("Dataset downloaded and saved.")
    return dataset


dataset = load_or_fetch_wikitext(dataset_name, dataset_config, dataset_path_head + dataset_filename)

print(dataset)


print(dataset['train'][:5])

print(dataset['train'])
print(dataset['train']['text'][:5])

# filtering out empty texts
def filter_empty_texts(examples):
    return bool(examples['text'].strip())


print("\nClean dataset does not exist and filtering empty text...")
for key in dataset.keys():
  print(f"Cleaning {key} dataset")
  dataset[key] = dataset[key].filter(filter_empty_texts)
  print(f"Finished cleaning {key} dataset")


# check results
print(f"\nChecking datasets")
for split, ds in dataset.items():
    print(f"{split} dataset: {len(ds)} rows after filtering")

# viewing first few elements
print("dataset['test']['text'][:5]: ", dataset['test']['text'][:5])
# 

# saving filtered text

print("\n### Create the directory to save if it does not exist ###")
# Create the directory if it does not exist
directory = home_dir + "/" + dataset_config + "-filtered-text-lists"
os.makedirs(directory, exist_ok=True)


key = "validation"
print("key: ", key)
filename = "/" + key + "_text_filtered.txt"
print("\n### saving to file ###")


# Using a unique delimiter to join and save strings
# Here we use uni_delim as it's unlikely to be in the text
with open(directory + filename, 'w', encoding='utf-8') as file:
    file.write(uni_delim.join(dataset[key]['text']))
print("\n### finished saving to file ###")

# 

# loading filtered text 

print("\n### loading from file ###")

# Load the strings back, splitting by the unique delimiter
with open(directory + filename, 'r', encoding='utf-8') as file:
    loaded_filtered_text = file.read().split(uni_delim)

print("\n### finished loading from file ###")


print("loaded_filtered_text[:5]: ", loaded_filtered_text[:5])

# checking if loaded filtered text matches filtered text computed
if loaded_filtered_text == dataset[key]['text']:
    print("The lists are the same.")
else:
    print("The lists are different.")



