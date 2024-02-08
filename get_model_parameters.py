import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, get_linear_schedule_with_warmup
from datasets import load_dataset, Dataset
import os
import pandas as pd
from tqdm import tqdm
import wandb
import psutil
import pytz
import json
import sys
from torch.optim import AdamW
import torch.nn.functional as func
from typing import List
import multiprocessing as mp
import math
import torch.nn as nn
# use dataloader for batching
from torch.utils.data import DataLoader, TensorDataset


print("torch.__version__: ", torch.__version__)
print("torch.version.cuda: ", torch.version.cuda)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# can double check if device is on gpu
print("GPU is available: ", torch.cuda.is_available()) # Should return True if CUDA is available and accessible by PyTorch
if torch.cuda.is_available():
    print("index of current gpu: ", torch.cuda.current_device())  # Returns the index of the current GPU
    print("number of gpus available to pytorch: ", torch.cuda.device_count())  # Returns the number of GPUs available to PyTorch
    print("name of device: ", torch.cuda.get_device_name(0))  # Returns the name of a device

#sys.exit()


print("\n### defining variables ###")
#### variables
# define training parameters
batch_size = 32 # adjust based on GPU memory
epochs = 3 # number of training epochs
learning_rate = 5e-4
batch_max = 3300

suffix = "-bs-" + str(batch_size) + "-e-" + str(epochs) + "-lr-" + format(learning_rate, '.0e') + "-bm-" + str(batch_max)

wandb_on = False
wandb_run_name = "val_loop" + suffix
wandb_project = "gpt-2-finetuning"


model_path = "./my_finetuned_gpt2" + suffix
model_layer_info_path = model_path + "-layer_info.txt"

max_len = 1024 # maximum sequence length, should be no larger than max context window of model (for gpt2, this is 1024)

uni_delim = "\0" # a unique delimiter to save and load filtered text from dataset
# Here we use "\0" (null character) as it's unlikely to be in the text

home_dir = os.getcwd()
dataset_config = 'wikitext-103-v1'


####


#### wandb setup

if wandb_on ==True:

    print("\n### setting up wandb ###")
    wandb.login()
    # initialize WandB
    wandb.init(project=wandb_project, name= wandb_run_name)

    print("\n### finished initializing wandb ###")
    # using wanb to log ram usage
    def log_ram_usage():
        # Get memory usage in percentage
        memory_usage_percent = psutil.virtual_memory().percent
        memory_usage_used = psutil.virtual_memory().used/(1024*1024*1024) # get memory used in gb
        memory_available = psutil.virtual_memory().available/(1024*1024*1024) # get memory used in gb; this is the memory that can be given instantly to processes without the system going into swap. 
        #print("memory_usage_used (GB): ", memory_usage_used)
        # Log the memory usage
        wandb.log({"RAM Usage (%)": memory_usage_percent, "RAM Usage (GB)": memory_usage_used, "RAM Available (GB)": memory_available})

#### finished wandb setup




#### printing variables
print("batch_size: ", batch_size)
print("epochs: ", epochs)
print("learning_rate: ", learning_rate)
print("suffix: ", suffix)

####

#### load a saved pretrained model

print("\n### now loading saved pretrained gpt2 model and tokenizer ### ")
# now load the pre-trained gpt2 model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_name) # gpt2 specific tokenizer, didn't save (because didn't change) original tokenizer
tokenizer_name = 'gpt2-tokenizer'

# Set the padding token (to existing special token, the end of sentence token)
tokenizer.pad_token = tokenizer.eos_token


# Get the total number of parameters
total_params = model.num_parameters()

# Get the number of trainable parameters
trainable_params = model.num_parameters(only_trainable=True)


with open(model_layer_info_path, 'w') as file:
    print(f"Total parameters: {total_params}")
    file.write(f"Total parameters: {total_params}\n")
    print(f"Trainable parameters: {trainable_params}") # I have 124439808 parameters (124M), so this is the gpt2-small
    file.write(f"Trainable parameters: {trainable_params}\n") 


    # Access the config to get the architecture details
    config = model.config

    print(f"Number of layers: {config.n_layer}")    
    file.write(f"Number of layers: {config.n_layer}\n")
    print(f"Number of attention heads: {config.n_head}")
    file.write(f"Number of attention heads: {config.n_head}\n")
    print(f"Size of embeddings: {config.n_embd}")
    file.write(f"Size of embeddings: {config.n_embd}\n")
    print(f"Maximum sequence length: {config.n_positions}")
    file.write(f"Maximum sequence length: {config.n_positions}\n")
    print(f"Vocabulary size: {config.vocab_size}")
    file.write(f"Vocabulary size: {config.vocab_size}\n")
    print(f"Total parameters: {model.num_parameters()}")
    file.write(f"Total parameters: {model.num_parameters()}\n")

    file.write("\n------\n")




#### 


with open(model_layer_info_path, 'a') as file:
    print(f"writing model layer info into file at {model_layer_info_path}")
# View parameters of the first transformer layer
    for name, param in model.named_parameters():
    # Write text to the file
        file.write(f"Parameter Name: {name}\n")
        file.write(f"Shape: {param.size()}\n")
        file.write(f"Requires Grad: {param.requires_grad}\n")
        file.write("------\n")
print("done writing")

print("exitting...")
sys.exit()


# code to freeze model parameters and add or finetune a single layer (such as the last)

# Freeze all parameters except for a specific layer (e.g., the last layer)
for parameter in model.parameters():
    parameter.requires_grad = False
for parameter in model.transformer.h[-1].parameters():  # Unfreeze the last layer
    parameter.requires_grad = True

# Add a custom layer and fine-tune it
# custom_layer = torch.nn.Linear(768, 768)
# model.transformer.h = torch.nn.ModuleList(list(model.transformer.h) + [custom_layer])

# one would add this before the training loop if you wanted to train the last layer






