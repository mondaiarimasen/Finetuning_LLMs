# module for all common methods for loading text
import wandb
import psutil
from typing import List, Dict, Union
import sys
import os 
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import math
from collections import deque
from glob import glob



#### checkpoints

# define checkpoints directory
def init_checkpoints_dir(checkpoint_dir):
    # Define the directory to save checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)

    
# returns list of all saved checkpoints
def get_all_checkpoints(checkpoint_dir: str) -> List:
    checkpoint_files = sorted(glob(os.path.join(checkpoint_dir, 'checkpoint_*.pt')))
    print("\ncheckpoint_files: ", checkpoint_files)
    return checkpoint_files

# save checkpoints, and make sure there are at most num_checkpoint number of checkpoints in directory at most; num_checkpoint is default 5
def save_checkpoint(checkpoint_dict: Dict, filename: str, num_checkpoint = 5):
    print("\nattempting to save checkpoint at ", filename)
    torch.save(checkpoint_dict, filename)
    print("successfully saved checkpoint_dict at ", filename)

    checkpoint_dir = os.path.dirname(filename)
    checkpoint_files = get_all_checkpoints(checkpoint_dir)

    if len(checkpoint_files) > num_checkpoint:
        oldest_checkpoint = checkpoint_files[0]
        print("attempting to remove oldest checkpoint from dir")
        try:
            os.remove(oldest_checkpoint)
            print("removed oldest checkpoint file from dir: ", oldest_checkpoint)
        except OSError:
            print("couldn't remove oldest_checkpoint")
            sys.exit()


# Function to load the latest checkpoint, returns the checkpoint or None
def load_latest_checkpoint(checkpoint_dir: str, device): #(model, optimizer):
    # Get all the checkpoint files and sort them
    checkpoint_files = get_all_checkpoints(checkpoint_dir)
    num_checkpoint_files = len(checkpoint_files)
    load_failed_bool = True
    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        print("latest_checkpoint: ", latest_checkpoint)
        i = num_checkpoint_files - 1
        while load_failed_bool and i>=0:
            try: 
                checkpoint = torch.load(checkpoint_files[i], map_location=device)
                print(f"finished loading checkpoint {i - num_checkpoint_files}")
                load_failed_bool = False
            except Exception as e:
                print(f"Failed to load checkpoint {i - num_checkpoint_files}: {e}")
                print("trying to load from next most checkpoint, if exists")
                i -=1
        
        # if all saved checkpoints failed, return None
        if load_failed_bool and i <0:
            return None
        # else return the checkpoint
        return checkpoint
    else:
        return None  # No checkpoint found, return none 





#### wandb setup

def init_wandb(project_name: str, run_name: str, resume: bool) -> None:
    print("\n### setting up wandb ###")
    wandb.login()
    # initialize WandB
    wandb.init(project=project_name, name= run_name, resume = resume)

    print("\n### finished initializing wandb ###")

# using wanb to log ram usage
def log_ram_usage():
    if wandb.run is not None: # check if wandb is logged in, otherwise do nothing
        # Get memory usage in percentage
        memory_usage_percent = psutil.virtual_memory().percent
        memory_usage_used = psutil.virtual_memory().used/(1024*1024*1024) # get memory used in gb
        memory_available = psutil.virtual_memory().available/(1024*1024*1024) # get memory used in gb; this is the memory that can be given instantly to processes without the system going into swap. 
        #print("memory_usage_used (GB): ", memory_usage_used)
        # Log the memory usage
        wandb.log({"RAM Usage (%)": memory_usage_percent, "RAM Usage (GB)": memory_usage_used, "RAM Available (GB)": memory_available})

# Define the (safe) logging function
def safe_wandb_log(metrics_dict):
    if wandb.run:
        wandb.log(metrics_dict)


#### finished wandb setup
    


#### loading saved filtered text for test, train, validation sets

# returns list with filtered key text
# uni_delim is a unique delimiter to save and load filtered text from dataset
# Here we use "\0" (null character) as it's unlikely to be in the text
def load_filtered_text(directory: str, key: str, uni_delim = "\0") -> List:
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


#### finished loading saved filtered text for test, train, validation sets





#### loading saved cleaned tokens for test, train, validation sets

# returns pytorch tensor of the cleaned tokens from specified file for the specified key
def load_clean_tokens(directory: str, key: str) -> Tensor:
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



#### finished loading saved cleaned tokens for test, train, validation sets


#### get cuda info
# returns device we are using; also naming first gpu we connected to as 'cuda:0'
def get_cuda_info():
    print("torch.__version__: ", torch.__version__)
    print("torch.version.cuda: ", torch.version.cuda)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # can double check if device is on gpu
    print("GPU is available: ", torch.cuda.is_available()) # Should return True if CUDA is available and accessible by PyTorch
    if torch.cuda.is_available():
        print("index of current gpu: ", torch.cuda.current_device())  # Returns the index of the current GPU
        print("number of gpus available to pytorch: ", torch.cuda.device_count())  # Returns the number of GPUs available to PyTorch
        print("name of device: ", torch.cuda.get_device_name(0))  # Returns the name of a device
    return device


#### use more gpus in parallel if possible


# returns the model (possibily as DataParallel object) ; and device that we are using
def use_more_gpus_in_parallel(model):
    print("checking if gpu device count is more than 1: ")
    # Then, use DataParallel to wrap your model if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        # Wrap the model with DataParallel
        model = nn.DataParallel(model)
        print("model is now output of DataParallel")
        # Move your model to the primary device
        #device = torch.device('cuda:0')
        print("finished moving to primary device cuda:0")
    else:
        print(f"Using {torch.cuda.device_count()} GPU")
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    return model#, device



#### get model parameters

def get_hl_model_info(model):
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


#### constructing tokenized_datasets_pt for a specified key

def get_tokenized_datasets_pt(keys: List[str], clean_token_dir: str, filtered_text_dir: str, max_length = 1024) -> Dict[str, Union[List[int], Tensor]]:
    print("\n### constructing tokenized_datasets_pt ###")

    # tokenized_datasets_pt is dict structure {'test' : {}, 'train': {}, 'validation': {}}
    # and for each key of tokenized_datasets_pt, tokenized_datasets_pt[key] 
    # is dict of structure {'text': original text, 'input_ids': 2d pytorch tensor where each row is len max_len, i.e. max seqence length, 'attention_mask': 2d pytorch tensor, same shape as input_ids, where 1 means corresponding element of input_ids is real data, 0 means it's a padding token}
    #keys = ['train'] #["test", "train", "validation"]
    tokenized_datasets_pt = {}
    for key in keys:
        print(f"getting text, input_ids, and attention_mask for {key}")
        key_clean_tokens = load_clean_tokens(clean_token_dir, key)

        # check to make sure the max sequence length is set properly
        if max_length!= key_clean_tokens.shape[1]:
            print("reshaping token tensor")
            key_clean_tokens = key_clean_tokens.view(-1,max_length)
            print("finished reshaping token tensor")

        tokenized_datasets_pt[key] = {'text': load_filtered_text(filtered_text_dir, key), 'input_ids': key_clean_tokens, 'attention_mask': torch.ones_like(key_clean_tokens)}
        # attention_mask is all 1s here because we don't use any padding in the tokenization (we just throw away the tail)

    print("\n### finished constructing tokenized_datasets_pt ###")
    return tokenized_datasets_pt


#### constructing tokenized_datasets_pt dict for test, train, and validation



#### data loading

# returns dataloader object with tokenized data in batches of size batch_size
def get_dataloader(batch_size: int, tokenized_datasets_pt: Dict): 

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


    print("dataloader['train']: ", dataloader['train'])


    print("\n### dataloader finished ###")

    return dataloader


#### saving pretrained and finetuned model

def save_finetuned_model(model, model_path: str):
    print("\nchecking if model is a DataParallel object")
    # Check if the model is a DataParallel object
    if isinstance(model, nn.DataParallel):
        print("it is, so saving with model.module.save_pretrained(model_path)")
        # If it is, save using the .module attribute
        model.module.save_pretrained(model_path)
    else:
        print("it's not, so saving with model.save_pretrained(model_path)")
        # If it's not, save it directly
        model.save_pretrained(model_path)
    print("finished saving")









