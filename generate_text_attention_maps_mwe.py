import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, get_linear_schedule_with_warmup
from datasets import load_dataset, Dataset
import os
from tqdm import tqdm
import pytz
import sys
from torch.optim import AdamW
from typing import List
import torch.nn as nn
from common_utils import *


device = get_cuda_info()


#sys.exit()

print("\n### defining variables ###")
#### variables
batch_size = 8 # adjust based on GPU memory
epochs = 10 # number of training epochs
learning_rate = 3e-4
batch_max = None #3300 # use None if want to run on all batches

max_len = 1024 # maximum sequence length, should be no larger than max context window of model (for gpt2, this is 1024)

suffix = "-bs-" + str(batch_size) + "-e-" + str(epochs) + "-lr-" + format(learning_rate, '.0e') + "-bm-" + str(batch_max) + "-ml-" + str(max_len)

start_epoch = 0

wandb_run_name = "training_loop" + suffix
wandb_project = "gpt-2-finetuning"
wandb_resume = True
wandb_on_bool = False

model_name = 'gpt2'
model_path = "./my_finetuned_" + model_name + suffix
save_model_bool = True

# uni_delim = "\0" # a unique delimiter to save and load filtered text from dataset
# Here we use "\0" (null character) as it's unlikely to be in the text

home_dir = os.getcwd()
dataset_config = 'wikitext-103-v1'

checkpoint_dir = 'checkpoints' + suffix
num_checkpoint = 1 # default is 5

#### printing variables
print("batch_size: ", batch_size)
print("epochs: ", epochs)
print("learning_rate: ", learning_rate)
print("suffix: ", suffix)

####


#### loading model

print("\n### now loading gpt2 model and tokenizer ### ")
# now load the pre-trained gpt2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name) # gpt2 specific tokenizer
tokenizer_name = 'gpt2-tokenizer'

# Set the padding token (to existing special token, the end of sentence token)
tokenizer.pad_token = tokenizer.eos_token

get_hl_model_info(model) # I have 124439808 parameters (124M), so this is the gpt2-small
model.to(device)

#### finished loading model


optimizer = AdamW(model.parameters(), lr = learning_rate)
# note that model.parameters() retrieves all the parameters (weights and biases)
# of the GPT-2 model. The optimizer needs these to know which values it should be updating during training.
print("\nlearning rate: ", learning_rate)


get_cuda_info()
torch.cuda.empty_cache()
print("emptied cuda cache")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# load latest checkpoint
checkpoint = load_latest_checkpoint(checkpoint_dir, device)
if checkpoint:
    print("\ncheckpoint exists and loading")
    start_epoch = checkpoint['next_epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("finished loading from checkpoint")
# if no checkpoint exists, then it uses the default settings



model = use_more_gpus_in_parallel(model)

#model.to(device) # move model to device

prompt = "The mysteries of the universe are"
generate_text(model, tokenizer, device, prompt, max_length = 100, num_beams = 5, num_return_sequences=5, early_stopping=True)


prompt = "To be honest,"
generate_text(model, tokenizer, device, prompt, max_length = 100, num_beams = 5, num_return_sequences=5, early_stopping=True)


prompt = "I am testing the GPT2 model which is"
generate_text(model, tokenizer, device, prompt, max_length = 100, num_beams = 5, num_return_sequences=5, early_stopping=True)

prompt = "I am testing the GPT2 model which is"
generate_text(model, tokenizer, device, prompt, max_length = 100, num_beams = 5, num_return_sequences=5, early_stopping=True)


input_text = "I am testing the GPT2 model."
layers = [0,1,11]
heads = [0,1,11]
draw_attention_map(model, tokenizer, device, input_text, layers, heads)



