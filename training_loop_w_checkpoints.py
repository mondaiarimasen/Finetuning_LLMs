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
epochs = 6 # number of training epochs
learning_rate = 5e-3
batch_max = 400 #3300 # use None if want to run on all batches

max_len = 1024 # maximum sequence length, should be no larger than max context window of model (for gpt2, this is 1024)

suffix = "-bs-" + str(batch_size) + "-e-" + str(epochs) + "-lr-" + format(learning_rate, '.0e') + "-bm-" + str(batch_max) + "-ml-" + str(max_len)

start_epoch = 0

wandb_run_name = "training_loop" + suffix
wandb_project = "gpt-2-finetuning"
wandb_resume = False
wandb_on_bool = True

model_name = 'gpt2'
model_path = "./my_finetuned_" + model_name + suffix
save_model_bool = False

# uni_delim = "\0" # a unique delimiter to save and load filtered text from dataset
# Here we use "\0" (null character) as it's unlikely to be in the text

home_dir = os.getcwd()
dataset_config = 'wikitext-103-v1'

checkpoint_dir = 'checkpoints' + suffix
num_checkpoint = 2 # default is 5

#### filtered text directory for test, train, validation sets
filtered_text_dir = home_dir + "/" + dataset_config + "-filtered-text-lists"
filtered_text_file_suffix = "_text_filtered.txt"

#### clean token dir for test, train, validation sets
clean_token_dir = home_dir + "/" + dataset_config + "-" + model_name + "-tokenizer-clean-tokens-pt"
clean_token_file_suffix = '_tokens_clean.pt'



####

#### checkpoints setup
init_checkpoints_dir(checkpoint_dir)

#### wandb setup
if wandb_on_bool:
    init_wandb(wandb_project, wandb_run_name, wandb_resume)
#### finished wandb setup



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


#### 
test_lst = load_filtered_text(filtered_text_dir + "/test" + filtered_text_file_suffix, "test")
test_tokens = load_clean_tokens(clean_token_dir + "/test" + clean_token_file_suffix, "test")
####

#### constructing tokenized_datasets_pt

keys = ['train', 'validation', 'test']
tokenized_datasets_pt = get_tokenized_datasets_pt(keys, clean_token_dir, clean_token_file_suffix, filtered_text_dir, filtered_text_file_suffix, max_length = max_len)
# tokenized_datasets_pt is dict structure {'test' : {}, 'train': {}, 'validation': {}}
# and for each key of tokenized_datasets_pt, tokenized_datasets_pt[key] 
# is dict of structure {'text': original text, 'input_ids': 2d pytorch tensor where each row is len max_len, i.e. max seqence length, 'attention_mask': 2d pytorch tensor, same shape as input_ids, where 1 means corresponding element of input_ids is real data, 0 means it's a padding token}

#### data loading

print("\n batch_size: ", batch_size)

dataloader = get_dataloader(batch_size, tokenized_datasets_pt) # this has the same keys as tokenized_datasets_pt

####

#print("dataloader['train']: ", dataloader['train'])

#sys.exit()


#### training 

print("-----starting training-----")


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

len_dataloader_train = len(dataloader['train'])
print("len_dataloader_train: ", len_dataloader_train)

len_dataloader_validation = len(dataloader['validation'])
print("len_dataloader_validation: ", len_dataloader_validation)

len_dataloader_test = len(dataloader['test'])
print("len_dataloader_test: ", len_dataloader_test)


batch_max = batch_max if batch_max is not None else len_dataloader_train

model.train() # tell model it's in training mode

# training loop
for epoch in range(start_epoch, epochs):
    log_ram_usage()
    total_loss = 0
    batch_num = 1
    #print("\nbatch_num: ", batch_num)
    print("\n")
    for batch in tqdm(dataloader['train'], desc=f"batch loop for train at epoch {epoch}/{epochs} (0-indexed)"):
        log_ram_usage()
        #print("in batch loop before if")
        if batch_num <= batch_max:
        
            #if batch_num == 1:
            #   print("in batch loop")
            
            # only need to move to device if I'm not using DataParallel
            #input_ids, attention_mask = [item.to(device) for item in batch]
            
            input_ids, attention_mask = [item for item in batch]
            input_ids = input_ids.long()
            attention_mask = attention_mask.long()

            input_ids.to(device)
            attention_mask.to(device)
            log_ram_usage()

            #print("input_ids.shape: ", input_ids.shape) 
            #print("type(input_ids): ", type(input_ids))
            #print("input_ids.dtype: ", input_ids.dtype)
            
            #print("computing labels")
            #labels = torch.cat((input_ids[:, 1:], torch.tensor([[-100]] * input_ids.size(0))), dim=1)
            # labels is the tensor of token indices (input_ids) shifted by 
            # one position to the left, with -100 appended to the end of 
            # each sequence to signal that there is no prediction to be made for the last token.
            # -100 is used because this is the ignore_index of cross entropy loss function (which is what outputs.loss computes) 
            # so any target label with the value -100 will not contribute to the loss, see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

            # need to make sure the tensors being passed into embedding 
            # layer are long or ints since in PyTorch, embedding layers 
            # are used to retrieve embeddings from an embedding matrix, 
            # and they require the indices to be integers because these 
            # indices are used to look up specific rows in the embedding matrix. 
            #labels = labels.long() 
            #labels.to(device)
            log_ram_usage()
            #print("feeding into model")
            
            # (overall idea: model performs a forward pass with the given inputs and calculates the loss
            # using the provided labels (which are input ids))
            # according to wandb, the following line takes around 20G RAM to run
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels = input_ids)
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
            
            total_loss += loss.item()
            
            log_ram_usage()
            # do backpropagation, computing the gradient of the loss with respect to each weight
            #print("doing backpropagation")
            loss.backward()
            log_ram_usage()

            '''
            # for debugging
            for param in model.parameters():
                print("checking model params on same device")
                assert param.device == torch.device('cuda:0'), "Model parameter device mismatch"

            for group in optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        print("checking device mismatch")
                        assert param.grad.device == param.device, "Gradient device mismatch"
            '''

            # optimizer updates the weights based on the gradients calculated during backpropagation
            #print("updating weights using optimizer")
            optimizer.step()
            # gradients are reset for the next batch
            #print("resetting gradients")
            optimizer.zero_grad()
            #print(f"done with batch {batch_num} in epoch {epoch}")
            #print(f"Epoch: {epoch}, Loss (in batch): {loss.item()}")
            batch_num+=1
            #wandb.log({"epoch": epoch, "loss": loss.item()})

            #else:
            #    print("breaking out of batch loop on batch ", batch_num)
            #    break
        else:
            print(f"reached batch_max of {batch_max}")
            break

    print("out of batch loop")

    avg_train_loss = total_loss / len_dataloader_train
    safe_wandb_log({"avg_train_loss": avg_train_loss})


    print(f"Epoch {epoch}/{epochs}, avg train loss: {avg_train_loss}")
    safe_wandb_log({"epoch": epoch}) 



    # Validation after each trainig epoch
    model.eval()
    total_eval_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader['validation'], desc=f"batch loop for validation at epoch {epoch}/{epochs} (0-indexed)"):
            input_ids, attention_mask = [item for item in batch]
            input_ids = input_ids.long()
            attention_mask = attention_mask.long()

            input_ids.to(device)
            attention_mask.to(device)
            log_ram_usage()

            #labels = torch.cat((input_ids[:, 1:], torch.tensor([[-100]] * input_ids.size(0))), dim=1)

            #labels = labels.long() 
            #labels.to(device)
            log_ram_usage()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels = input_ids)
            loss = outputs.loss
            
            #print("taking mean of outputs if outputs.ndim>0, else just returning outputs")
            loss = loss.mean() if loss.ndim > 0 else loss

            total_eval_loss += loss.item()

            log_ram_usage()
    
    # Calculate average loss over the validation data
    avg_val_loss = total_eval_loss / len_dataloader_validation
    print(f'Epoch {epoch} validation loss: {avg_val_loss}')
    safe_wandb_log({"avg_val_loss": avg_val_loss}) 

    # Calculate the perplexity based on the mean validation loss
    validation_perplexity = torch.exp(torch.tensor(avg_val_loss))
    print(f'Epoch {epoch} validation perplexity: {validation_perplexity}')
    safe_wandb_log({"validation perplexity": validation_perplexity}) 


    # Reset model to training mode
    model.train()

    # Additional information you might want to save with the model
    checkpoint_dict_to_save = {
        'next_epoch': epoch+1,
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'avg_train_loss': avg_train_loss,
        'avg_val_loss': avg_val_loss,
        # Include any other data you need to resume training
    }

    # Save the checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch:04d}.pt')
    save_checkpoint(checkpoint_dict_to_save, checkpoint_path, num_checkpoint = num_checkpoint)
    print(f"epoch {epoch} and checkpoint files are {get_all_checkpoints(checkpoint_dir)}")
    



print("-----finished training and validation-----")

#print("finishing wandb")
#wandb.finish()

#sys.exit()

#### 

#### saving fine tuned model

# save the fine tuned model
# from google.colab import drive
# drive.mount('/content/drive')

# model.save_pretrained('/content/drive/My Drive/my_finetuned_model')
# tokenizer.save_pretrained('/content/drive/My Drive/my_finetuned_model')

if save_model_bool: 
    save_finetuned_model(model, model_path)

#tokenizer.save_pretrained(model_path) # don't need to save tokenizer because it's unchanged

#### testing finetuned model

# test fine-tuned model
prompt = "The mysteries of the universe are"
inputs = tokenizer.encode(prompt, return_tensors = 'pt').to(device)
# Create an attention mask for the inputs
attention_mask = torch.ones(inputs.shape, dtype=torch.long, device=device)
# Set pad_token_id to the pad_token_id of the tokenizer
pad_token_id = tokenizer.pad_token_id


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
    attention_mask=attention_mask,
    pad_token_id=pad_token_id,
    max_length=100, 
    num_beams=5, 
    num_return_sequences=5, 
    early_stopping=True # Stop generating once max_length is reached
)

# note that this max_length doesn't need to be the same as the max_len in the tokenization process of train data; they are independent
# the max_length here is just the max length of the generated sequence that I'm outputing here

print("Generated text:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))



#### testing on test set
print("\ntesting on test set")
model.eval()
total_test_loss = 0
with torch.no_grad():
    for batch in tqdm(dataloader['test'], desc=f"batch loop for test"):
        input_ids, attention_mask = [item for item in batch]
        input_ids = input_ids.long()
        attention_mask = attention_mask.long()

        input_ids.to(device)
        attention_mask.to(device)

        log_ram_usage()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels = input_ids)
        loss = outputs.loss
        
        #print("taking mean of outputs if outputs.ndim>0, else just returning outputs")
        loss = loss.mean() if loss.ndim > 0 else loss

        total_test_loss += loss.item()

        log_ram_usage()

# Calculate average loss over the validation data
avg_test_loss = total_test_loss / len_dataloader_test
print(f'test loss: {avg_test_loss}')
safe_wandb_log({"avg_test_loss": avg_test_loss}) 

# Calculate the perplexity based on the mean validation loss
test_perplexity = torch.exp(torch.tensor(avg_test_loss))
print(f'Test perplexity: {test_perplexity}')
safe_wandb_log({"test perplexity": test_perplexity}) 













wandb.finish()
sys.exit()




# evaluate on test data
total_test_loss = 0
total_length = 0

# some sample code
for batch in tokenized_dataset["test"]:
  inputs = {k:v.to(device) for k,v in batch.items()}
  with torch.no_grad():
    outputs = model(**inputs, labels = inputs["input_ids"])
    loss = outputs.loss
    total_test_loss += loss.item() * inputs["input_ids"].size(1)
    total_length += inputs["input_ids"].size(1)


avg_test_loss = total_test_loss / total_length
test_perplexity = np.exp(avg_test_loss)

print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Perplexity: {test_perplexity:.2f}")




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




