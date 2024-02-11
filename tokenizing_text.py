import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, GPT2Config, get_linear_schedule_with_warmup, AutoTokenizer, TextDataset
from tqdm import tqdm
import sys
from typing import List
from cleaning_data_utils import *
from tokenizing_utils import *


### variables
keys = ["test", "validation", "train"]
uni_delim = "\0"

wandb_run_name = "attempting tokenizing test[:]"
wandb_project = "gpt-2-finetuning"

dataset_name = 'wikitext'
dataset_config = 'wikitext-103-v1'
dataset_path_head = os.getcwd()
home_dir = os.getcwd()
dataset_filename = '/wikitext-103-raw-v1'

dataset = load_or_fetch_wikitext(dataset_name, dataset_config, dataset_path_head + dataset_filename)

clean_dataset_dir = home_dir + "/" + dataset_config + "-filtered-text-lists"


###


# Example usage
clean_dataset = {}

for key in keys:
  clean_dataset[key] = {}

  clean_key_dataset_filename = "/" + key + "_text_filtered.txt"
  clean_dataset_path = clean_dataset_dir + clean_key_dataset_filename

  print(f"\nattempting to load clean {key} dataset")
  if os.path.exists(clean_dataset_path): # if clean dataset already exists
    print(f"Clean {key} dataset exists and loading...")
    clean_dataset[key]['text'] = load_clean_dataset(clean_dataset_path, uni_delim = uni_delim)
    print(f"Finished loading clean {key} dataset")
    print(dataset)
  else:
    print(f"\nClean {key} dataset does not exist and exiting...")
    print("you should clean the data first using the saving and checking filter script")
    sys.exit()
    

# check results
print(f"\nChecking datasets")
for split, ds in clean_dataset.items():
  print(f"{split} dataset: {len(ds['text'])} rows after filtering")

# viewing first few elements
print(clean_dataset['train']['text'][:5])
print(clean_dataset['validation']['text'][:5])
print(clean_dataset['test']['text'][:5])



print("\n### now loading gpt2 model and tokenizer ### ")
# now load the pre-trained gpt2 model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name) # gpt2 specific tokenizer
tokenizer_name = 'gpt2-tokenizer'

# Set the padding token (to existing special token, the end of sentence token)
tokenizer.pad_token = tokenizer.eos_token

max_len = 1024 # maximum sequence length, should be no larger than max context window of model (for gpt2, this is 1024)


#### sanity check for tokenization
sanity_check_tokenize_idpt_splice(clean_dataset['train']['text'], tokenizer, step = 10)
####

sys.exit()







print("\n### testing methods to convert list of tokens to pytorch tensor ###")
## this cell is to test the methods needed to convert the list of tokens
## in separate pytorch tensors into one big pytorch tensor with num of
## elements in each row as max_len


# tokenize all the text by doing it every few elements
# below two lines are for testing
test_list = [torch.tensor([[1]]),torch.tensor([[2,3]]),torch.tensor([[4, 5, 6]]),torch.tensor([[7,8,9,10]]),torch.tensor([[11,12,13,14,15]]),torch.tensor([[16, 17,18,19,20,21]]), torch.tensor([[22,23]])]


# now create the tokenized dataset with max_len max_sequence_length
# constructing the m * max_len 2d array of tokens; m is ceiling of len of
# {string of all text} divided by max_len; we pad the tail
def separate_and_reshape(l1, max_len):
    len_l1 = len(l1)
    min_size = len_l1 // max_len

    # Separate the first 2 * max_len elements
    first_part = l1[:min_size * max_len]

    # Reshape the first part into a 2D array
    first_part_reshaped = first_part.view(-1, max_len)

    # Get the remaining elements
    if len(l1[min_size * max_len:]) == 0:
      remaining_part = l1[min_size * max_len:]
    else:
      remaining_part = l1[min_size * max_len:].view(-1,len_l1-min_size * max_len)

    return first_part_reshaped, remaining_part

# returns the tokens array of size m * max_len 2d array of tokens; m is
# ceiling of len of {string of all text} divided by max_len; we pad the tail if specified (otherwise don't keep it)
def get_tokens_array_clean(tokens_list: List[torch.Tensor], max_length: int = 1024, padding: bool = True, padding_token = None) -> torch.Tensor:
  if padding and padding_token is None:
    raise ValueError("Specify padding token if want to pad")

  # make copy of token list since don't want to modify original
  tokens_list = tokens_list.copy()

  # initialize final tensor
  tokens_pt_final = torch.empty(0)
  len_list = len(tokens_list)
  print(tokens_list)
  print(len_list)

  progress_bar = tqdm(total=len_list, desc="reshaping, padding, returning cleaned tensor")
  i=0
  while i < len_list:
    log_ram_usage()
    #print(i)
    curr = tokens_list[i]
    # if len of curr token tensor is less than max_len
    if curr.size(1) < max_length:
      # if at last element of list
      if i == len_list - 1:
        # check if need to pad
        if padding:
          curr = func.pad(curr, (0, max_length - curr.size(1)), 'constant', padding_token)
          tokens_pt_final = torch.cat((tokens_pt_final,curr),dim=0)
          return tokens_pt_final
        else:
          #print(f"{i} here, done")
          return tokens_pt_final
      else: # if not at last element of list
        i+=1
        #print("in if-else")
        progress_bar.update(1)
        # append the next element to lengthen the current token tensor
        curr = torch.cat((curr,tokens_list[i]), dim=1)
        tokens_list[i] = curr
        #print(f"{i} if ", curr)
    else: # reshape and save remainder in current index of tokens list
      reshaped_part, remaining_part = separate_and_reshape(curr[0],max_length)
      #print(f"{i} else ", reshaped_part, remaining_part)
      tokens_pt_final = torch.cat((tokens_pt_final,reshaped_part),dim=0)
      #print(f"{i} else tokens_pt_final", tokens_pt_final)
      tokens_list[i] = remaining_part
      # if there is no remaining part, go to next element in tokens list
      if remaining_part.size(0) == 0:
        i+=1
        #print("remaining part is 0 size")
        progress_bar.update(1)

  progress_bar.close()

  return tokens_pt_final

max_lensdf = 6
print(get_tokens_array_clean(test_list, max_length = max_lensdf, padding = True, padding_token = 100))

print(get_tokens_array_clean(test_list, max_length = max_lensdf, padding = False))



# after sanity check passes, tokenizing for real
print("\n### tokenizing test set for real ###")

# list of pt tensors of tokenized strings (step number at a time) in test dataset
tokens_test_pt_list = []

for i in tqdm(range(0, len(dataset['test']['text']), step), desc=f"tokenizing test dataset"):
  tokens_test_pt_list.append(tokenizer.encode(''.join(dataset['test']['text'][i:i+step]), return_tensors='pt'))


print(tokens_test_pt_list[0])
print(len(tokens_test_pt_list))

print(len(tokens_test_pt_list))



print("\n### getting final token tensor for test data ###")

## getting final token tensor for test data; note that these are the input_ids; 
## the attention_masks is a tensor of the same shape but all 1s (meaning all the data is real data, none of it is a padding token)
test_tokens_clean = get_tokens_array_clean(tokens_test_pt_list, max_length = max_len, padding = False)
print(test_tokens_clean.size())
print(f"total of {test_tokens_clean.size(0) * test_tokens_clean.size(1)} tokens for test")


print("\n### Create the directory to save if it does not exist ###")
# Create the directory if it does not exist
directory = home_dir + "/" + dataset_config + "-" + model_name + "-tokenizer-clean-tokens-pt"
os.makedirs(directory, exist_ok=True)

print("\n### saving test_tokens_clean ###")
# saving test_tokens_clean
torch.save(test_tokens_clean, directory + '/test_tokens_clean.pt')

print("\n### checking if saved correctly ###")
# checking if saved correctly
loaded_tensor = torch.load(directory + '/test_tokens_clean.pt')
print(loaded_tensor)

are_same = torch.equal(test_tokens_clean, loaded_tensor)
print("loaded tensor and computed tensor are same: ", are_same)  # Output: True

# Check if the file exists
if os.path.exists(directory + '/test_tokens_clean.pt'):
    print("File exists.")
else:
    print("File does not exist.")

'''

print("\n### tokenizing validation set for real ###")
token_key = "validation"
# tokenizing validation data
# list of pt tensors of tokenized strings (step number at a time) in validation dataset
tokens_val_pt_list = []

for i in tqdm(range(0, len(dataset[token_key]['text']), step), desc=f"tokenizing {token_key} dataset"):
  tokens_val_pt_list.append(tokenizer.encode(''.join(dataset[token_key]['text'][i:i+step]), return_tensors='pt'))


print(tokens_val_pt_list[0])
print(len(tokens_val_pt_list))


print(max_len)

print(f"\n### getting final token tensor for {token_key} data ###")
## getting final token tensor for validation data
val_tokens_clean = get_tokens_array_clean(tokens_val_pt_list, max_length = max_len, padding = False)
print(val_tokens_clean.size())
print(f"total of {val_tokens_clean.size(0) * val_tokens_clean.size(1)} tokens for validation set")


print("\n### Create the directory to save if it does not exist ###")
# Create the directory if it does not exist
directory = home_dir + "/" + dataset_config + "-" + model_name + "-tokenizer-clean-tokens-pt"
os.makedirs(directory, exist_ok=True)

print(f"\n### saving {token_key}_tokens_clean ###")
# saving test_tokens_clean
torch.save(val_tokens_clean, directory + '/' + token_key + '_tokens_clean.pt')

print("\n### checking if saved correctly ###")
# checking if saved correctly
loaded_tensor = torch.load(directory + '/' + token_key + '_tokens_clean.pt')
print(loaded_tensor)

are_same = torch.equal(val_tokens_clean, loaded_tensor)
print("loaded tensor and computed tensor are same: ", are_same)  # Output: True

# Check if the file exists
if os.path.exists(directory + '/' + token_key + '_tokens_clean.pt'):
    print("File exists.")
else:
    print("File does not exist.")



print("max_len: ", max_len)
wandb.log({"max_len ": max_len})


print("\n### tokenizing train set for real ###")

print("\n### attempting multiprocessing tokenization###")
def tokenize_data(data_chunk):
    # using the GPT2 tokenizer here!
    return tokenizer.encode(''.join(data_chunk), return_tensors='pt')

def process_in_chunks(dataset, chunk_size=10000):
    pool = mp.Pool(mp.cpu_count())
    results = []
    print("chunk_size: ", chunk_size)
    wandb.log({"chunk_size in mp: ": chunk_size})

    for i in tqdm(range(0, len(dataset), chunk_size), desc=f"tokenizing train dataset, mp"):
        chunk = dataset[i:i + chunk_size]
        result = pool.apply_async(tokenize_data, [chunk])
        results.append(result)
        log_ram_usage()

    pool.close()
    pool.join()

    print("type(result.get()): ", type(results[0].get()))
    return [result.get() for result in tqdm(results, desc = "getting results")]

# Load your dataset, using all of train data
dataset_text = dataset['train']['text']

# Process in chunks
tokens_train_pt_list = process_in_chunks(dataset_text, chunk_size = 1000)

# Now, tokenized_chunks contains the tokenized data



print("len(tokens_train_pt_list): ", len(tokens_train_pt_list))
print("tokens_train_pt_list[0]: ", tokens_train_pt_list[0])

print("\n### getting final token tensor for train data ###")
## getting final token tensor for train data
train_tokens_clean = get_tokens_array_clean(tokens_train_pt_list, max_length = max_len, padding = False)
print("train_tokens_clean.size(): ", train_tokens_clean.size())
wandb.log({"train_tokens_clean.size(): ": train_tokens_clean.size()})
print(f"total of {train_tokens_clean.size(0) * train_tokens_clean.size(1)} tokens for train set")
wandb.log({"total tokens for train set ": train_tokens_clean.size(0) * train_tokens_clean.size(1)})

print("\n### Create the directory to save if it does not exist ###")
# Create the directory if it does not exist
directory = home_dir + "/" + dataset_config + "-" + model_name + "-tokenizer-clean-tokens-pt"
os.makedirs(directory, exist_ok=True)

# saving test_tokens_clean
save_file = directory + '/train_tokens_clean.pt'
torch.save(train_tokens_clean, save_file)
wandb.log({"saving all train tokens to ": save_file})

# checking if saved correctly
loaded_tensor = torch.load(save_file)
print(loaded_tensor)

are_same = torch.equal(train_tokens_clean, loaded_tensor)
print("loaded tensor and computed tensor are same: ", are_same)  # Output: True

# Check if the file exists
if os.path.exists(save_file):
    print("File exists.")
else:
    print("File does not exist.")
'''




# Finish the run
wandb.finish()


