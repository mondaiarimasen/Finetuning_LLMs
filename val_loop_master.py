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

wandb_on = True
wandb_run_name = "val_loop" + suffix
wandb_project = "gpt-2-finetuning"


model_path = "./my_finetuned_gpt2" + suffix

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

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}") # I have 124439808 parameters (124M), so this is the gpt2-small

# Access the config to get the architecture details
config = model.config

print(f"Number of layers: {config.n_layer}")
print(f"Number of attention heads: {config.n_head}")
print(f"Size of embeddings: {config.n_embd}")
print(f"Maximum sequence length: {config.n_positions}")
print(f"Vocabulary size: {config.vocab_size}")
print(f"Total parameters: {model.num_parameters()}")



#### 


#### Using saved pretrained model on some sample text


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# can double check if device is on gpu
print("GPU is available: ", torch.cuda.is_available()) # Should return True if CUDA is available and accessible by PyTorch
if torch.cuda.is_available():
    print("index of current gpu: ", torch.cuda.current_device())  # Returns the index of the current GPU
    print("number of gpus available to pytorch: ", torch.cuda.device_count())  # Returns the number of GPUs available to PyTorch
    print("name of device: ", torch.cuda.get_device_name(0))  # Returns the name of a device




print("checking if gpu device count is more than 1: ")
# Then, use DataParallel to wrap your model if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    # Wrap the model with DataParallel
    model = nn.DataParallel(model)
    print("model is now output of DataParallel")
    # Move your model to the primary device
    device = torch.device('cuda:0')
    print("finished moving to primary device cuda:0")
else:
    print(f"Using {torch.cuda.device_count()} GPU")



model.to(device)




# test fine-tuned model
prompt = "Tell me a story"
inputs = tokenizer.encode(prompt, return_tensors = 'pt').to(device)

print("\nchecking if model is a DataParallel object")
# Check if the model is a DataParallel object
if isinstance(model, nn.DataParallel):
    print("it is, so using model.module to generate")
    # If it is, use model.module 
    model = model.module    
else:
    print("it's not, so using model to generate")

print("\ngenerating output")
outputs = model.generate(
    inputs, 
    max_length=100, 
    num_beams=5, 
    num_return_sequences=5, 
    early_stopping=True # Stop generating once max_length is reached
)

# note that this max_length doesn't need to be the same as the max_len in the tokenization process of train data; they are independent
# the max_length here is just the max length of the generated sequence that I'm outputing here


print("Generated text:")
for i in range(5):
    print(tokenizer.decode(outputs[i], skip_special_tokens=True))




####






#sys.exit()











#### loading saved filtered text for test, train, validation sets


filtered_text_dir = home_dir + "/" + dataset_config + "-" + model_name + "-filtered-text-lists"

# returns list with filtered key text
def load_filtered_text(directory, key):
    filename = directory + "/" + key + "_text_filtered.txt"
    # check if the filtered dataset txt file exists
    if os.path.exists(filename): # if filtered dataset txt file for that key already exists
        print(f"\nFiltered {key} text dataset exists and loading...")
        # Load the strings, splitting by the unique delimiter
        with open(filename, 'r', encoding='utf-8') as file:
            loaded_lst = file.read().split(uni_delim)

        print(f"Finished loading filtered {key} dataset")
        print(f"filtered {key} text, len(loaded_lst): ", len(loaded_lst))
        # for wikitext-103-raw-v1 dataset, len of filtered texts are as follows: test = 2891, train = 1165029, validation = 2461 

    else:
        print(f"\nfiltered {key} text dataset does not exist and ending...")
        sys.exit()
        
    return loaded_lst


# loading filtered text 




#### creating input_id and attention_masks tensors 

# loading saved cleaned tokens for test, train, validation sets

clean_token_dir = home_dir + "/" + dataset_config + "-" + model_name + "-tokenizer-clean-tokens-pt"

# returns pytorch tensor of the cleaned tokens from specified file
def load_clean_tokens(directory, key):
    filename = directory + "/" + key + '_tokens_clean.pt'
    # check if the clean tokens txt file exists
    if os.path.exists(filename): # if clean tokens txt file for that key already exists
        print(f"\nClean {key} tokens pt file exists and loading...")
        # Load the strings, splitting by the unique delimiter
        loaded_tensor = torch.load(filename)

        print(f"Finished loading clean {key} tokens")
        print(f"clean {key} tokens shape: ", loaded_tensor.shape)
        # for wikitext-103-raw-v1 dataset, shape of cleaned tokens (no padding, max sequence length is max_len=1024) is as follows:
        # test = torch.Size([275, 1024]), train = torch.Size([115039, 1024]), validation = torch.Size([241, 1024])

    else:
        print(f"\nClean {key} tokens pt file does not exist and ending...")
        sys.exit()
        
    return loaded_tensor




#### 



#### constructing tokenized_datasets_pt

print("\n### constructing tokenized_datasets_pt ###")

# tokenized_datasets_pt is dict structure {'test' : {}, 'train': {}, 'validation': {}}
# and for each key of tokenized_datasets_pt, tokenized_datasets_pt[key] 
# is dict of structure {'text': original text, 'input_ids': 2d pytorch tensor where each row is len max_len, i.e. max seqence length, 'attention_mask': 2d pytorch tensor, same shape as input_ids, where 1 means corresponding element of input_ids is real data, 0 means it's a padding token}
keys = ['validation'] #["test", "train", "validation"]
tokenized_datasets_pt = {}
for key in keys:
    print(f"getting text, input_ids, and attention_mask for {key}")
    key_clean_tokens = load_clean_tokens(clean_token_dir, key)
    tokenized_datasets_pt[key] = {'text': load_filtered_text(filtered_text_dir, key), 'input_ids': key_clean_tokens, 'attention_mask': torch.ones_like(key_clean_tokens)}
    # attention_mask is all 1s here because we don't use any padding in the tokenization (we just throw away the tail)


print("\n### finished constructing tokenized_datasets_pt ###")


#### data loading


dataloader = {}

dataset = {}

print("\n batch_size: ", batch_size)

print("\n### for loop for dataloading ###")
# for loop for dataloading
for key in tokenized_datasets_pt.keys(): 

    print(f"\ncreating tensor dataset for {key}") # doing this because dataloader expects dataset object as argument
    dataset[key] = TensorDataset(
        tokenized_datasets_pt[key]['input_ids'],
        tokenized_datasets_pt[key]['attention_mask']
    )
    
    print(f"\ncreating dataloader for {key}")
    dataloader[key] = DataLoader(dataset[key], batch_size=batch_size, shuffle=True)
    # checking number of batches for key = len(tokenized_datasets['key']) / batch_size
    print(f"checking expected number of {key} batches")
    if len(dataloader[key]) == math.ceil(len(tokenized_datasets_pt[key]['input_ids']) / batch_size):
        print(f"expected number of {key} batches is correct")
        
        print("len(dataloader[key]) :", len(dataloader[key]))
        print("len(tokenized_datasets_pt[key]['input_ids']): ", len(tokenized_datasets_pt[key]['input_ids']))
        print("batch_size: ", batch_size)
        
        # To view the contents of the DataLoader:
        for i, data in enumerate(dataloader[key]):
            print(f"Batch {i}:")
            print(data)
            print("first value should be equal to batch_size: ", data[0].shape, data[0].shape[0] == batch_size)
            # If you only want to see the first few batches, you can break early
            if i == 1:  # Adjust the number as needed
                break
        

    else:
        print(f"expected number of {key} batches is not correct")
        print("len(dataloader[key]) :", len(dataloader[key]))
        print("len(tokenized_datasets_pt[key]['input_ids']): ", len(tokenized_datasets_pt[key]['input_ids']))
        print("batch_size: ", batch_size)
        
        # To view the contents of the DataLoader:
        for i, data in enumerate(dataloader[key]):
            print(f"Batch {i}:")
            print(data)
            print("first value should be equal to batch_size: ", data[0].shape, data[0].shape[0] == batch_size)
            # If you only want to see the first few batches, you can break early
            if i == 1:  # Adjust the number as needed
                break
        
        sys.exit()

print("\n### finished for loop for dataloading ###")
# for any key in keys = ["test", "train", "validation"], a single item 
# in dataloader[key] is of the structure [batch_size * max_len pt tensor, batch_size * max_len pt tensor],
# where the first element is from 'input_ids' and the second element is from 'attention_mask'



#sys.exit()




print("dataloader['validation']: ", dataloader['validation'])


print("\n### dataloader finished ###")


















#### assessing saved pretrained model on validation dataset


print("-----starting applying to validation data-----")


optimizer = AdamW(model.parameters(), lr = learning_rate)
# note that model.parameters() retrieves all the parameters (weights and biases)
# of the GPT-2 model. The optimizer needs these to know which values it should be updating during training.
print("\nlearning rate: ", learning_rate)

print("torch.__version__: ", torch.__version__)
print("torch.version.cuda: ", torch.version.cuda)

# train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# can double check if device is on gpu
print("GPU is available: ", torch.cuda.is_available()) # Should return True if CUDA is available and accessible by PyTorch
if torch.cuda.is_available():
    print("index of current gpu: ", torch.cuda.current_device())  # Returns the index of the current GPU
    print("number of gpus available to pytorch: ", torch.cuda.device_count())  # Returns the number of GPUs available to PyTorch
    print("name of device: ", torch.cuda.get_device_name(0))  # Returns the name of a device




print("checking if gpu device count is more than 1: ")
# Then, use DataParallel to wrap your model if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    # Wrap the model with DataParallel
    model = nn.DataParallel(model)
    print("model is now output of DataParallel")
    # Move your model to the primary device
    device = torch.device('cuda:0')
    print("finished moving to primary device cuda:0")
else:
    print(f"Using {torch.cuda.device_count()} GPU")



model.to(device) # move model to device
#print("model.device: ", model.device)



model.eval() # tell model it's in evaluation mode



# validation loop
with torch.no_grad():
    log_ram_usage()
    total_loss = 0
    batch_num = 1
    #print("\nbatch_num: ", batch_num)
    for batch in tqdm(dataloader['validation'], desc=f"batch loop for validation"):#tokenized_datasets["train"]:
        log_ram_usage()
        #print("in batch loop before if")
        #if batch_num <= batch_max:
            
        #if batch_num == 1:
        #   print("in batch loop")
        
        # only need to move to device if I'm not using DataParallel
        #input_ids, attention_mask = [item.to(device) for item in batch]
        
        input_ids, attention_mask = [item for item in batch]
        input_ids = input_ids.long()
        attention_mask = attention_mask.long()

        log_ram_usage()

        #print("\nmoving input_ids and attention_mask to device")
        input_ids.to(device)
        attention_mask.to(device)

        # tokenizer(batch["text"]): tokenize text data in batch:
        # return_tensors = "pt": return the tokens as pytorch tensors ("pt" stands for pytorch, "tf" stands for tensorflow)
        # padding = True: ensures all tokenized sequences in batch have same length by padding shorter sequences with a special token
        # (need to pad when batching together sequences of different lengths so that they can be represented as a single tensor)
        # truncation = True: truncates the sequences to a max_length; it's important since gpt2 has a fixed length input size it can handle
        #inputs = tokenizer(batch["text"],return_tensors = "pt",padding = True, truncation = True, max_length = 512)
        
        #print("input_ids.shape: ", input_ids.shape) 
        #print("type(input_ids): ", type(input_ids))
        #print("input_ids.dtype: ", input_ids.dtype)
        

        # iterates through key-value pairs in inputs dictionary
        # moves value (which is a tensor) to same device as the model to so model can process the inputs
        # (will get an error if inputs and model are on different devices)
        #inputs = {k: v.to(device) for k,v in inputs.items()}

        #inputs = {k: v.to(device) for k,v in batch.items()}
        #print("computing labels")
        labels = torch.cat((input_ids[:, 1:], torch.tensor([[-100]] * input_ids.size(0))), dim=1)
        # need to make sure the tensors being passed into embedding 
        # layer are long or ints since in PyTorch, embedding layers 
        # are used to retrieve embeddings from an embedding matrix, 
        # and they require the indices to be integers because these 
        # indices are used to look up specific rows in the embedding matrix. 
        labels = labels.long() 
        #print("labels.size: ", labels.shape)
        #print("type(labels): ", type(labels))
        #print("labels.dtype: ", labels.dtype)
        # labels is the tensor of token indices (input_ids) shifted by 
        # one position to the left, with -100 appended to the end of 
        # each sequence to signal that there is no prediction to be made for the last token.
        # -100 is used because this is the ignore_index of cross entropy loss function (which is what outputs.loss computes) 
        # so any target label with the value -100 will not contribute to the loss, see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html


        labels.to(device)

        # passes tokenized and formatted inputs into model
        # **inputs: unpacks the dictionary into keyword arguments, i.e. passing all items in
        # inputs dictionary (like input ids, attention masks, etc.) as separate arguments to the model
        # labels = inputs["input_ids"]: for gpt2, labels are usually the same as input ids, just shifted.
        # by predicting the next token in the sequence, the model learns to generate text.
        # here, we are explicitly providing these input IDs as labels for the model to compare its predictions against.
        # (overall idea: model performs a forward pass with the given inputs and calculates the loss
        # using the provided labels (which are input ids))
        #outputs = model(**inputs)#, labels = inputs["input_ids"])
        log_ram_usage()
        #print("feeding into model")
        
        # according to wandb, the following line takes around 20G RAM to run
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels = labels)
        log_ram_usage()

        # output from model is a complex object containing various items, one of which is the loss,
        # representing how far off the model's predictions were from the actual values (the labels)
        #print("computing outputs.loss")
        loss = outputs.loss

        # checking loss shape since it might be a vector of length the number of gpus i'm using
        #print("loss.shape: ", loss.shape)
        #print("loss: ", loss)

        #print("taking mean of outputs if outputs.ndim>0, else just returning outputs")
        loss = loss.mean() if loss.ndim > 0 else loss


        #print("loss.shape: ", loss.shape)
        #print("loss: ", loss)

        # add loss for this batch to total_loss
        total_loss += loss.item()
        
        log_ram_usage()
        
        #else:
        #    print("breaking out of batch loop on batch ", batch_num)
        #    break
        #else:
        #    print(f"reached batch_max of {batch_max}")
        #    break

    print("out of batch loop")

    avg_val_loss = total_loss / len(dataloader['validation'])
    wandb.log({"avg_val_loss": avg_val_loss})


    print(f"Total Loss: {total_loss}, avg val loss: {avg_val_loss}")
    wandb.log({"total loss per epoch": total_loss})

print("-----finished assessing on validation data-----")

#print("finishing wandb")
#wandb.finish()

#sys.exit()


perplexity = torch.exp(torch.tensor(avg_val_loss))

print(f'Average Loss: {avg_val_loss}')
print(f'Perplexity: {perplexity}')


print("finishing wandb")
wandb.finish()











