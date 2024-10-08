from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, GPT2Config, get_linear_schedule_with_warmup, AutoTokenizer, TextDataset
import sys
from cleaning_data_utils import *
from tokenizing_utils import *
from common_utils import *

### variables
keys = ["test", "validation", "train"]
uni_delim = "\0"
step = 10
max_len = 1024 # maximum sequence length, should be no larger than max context window of model (for gpt2, this is 1024)
pad_tokens_bool = False
return_tensors = 'pt'

wandb_run_name = "attempting tokenizing test[:]"
wandb_project = "gpt-2-finetuning"

dataset_name = 'wikitext'
dataset_config = 'wikitext-103-v1'
dataset_path_head = os.getcwd()
home_dir = os.getcwd()
dataset_filename = '/wikitext-103-raw-v1'

clean_dataset_dir = home_dir + "/" + dataset_config + "-filtered-text-lists"

model_name = 'gpt2'
clean_token_dir = home_dir + "/" + dataset_config + "-" + model_name + "-tokenizer-clean-tokens-pt-correct"


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
    print(len(clean_dataset[key]['text']))
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



#### now load the pre-trained gpt2 model and tokenizer
print(f"\n### now loading {model_name} model and tokenizer ### ")
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name) # gpt2 specific tokenizer
tokenizer_name = 'gpt2-tokenizer'

# Set the padding token (to existing special token, the end of sentence token)
tokenizer.pad_token = tokenizer.eos_token

####


#### sanity check for tokenization
sanity_check_tokenize_idpt_splice(clean_dataset['train']['text'], tokenizer, step = 10)
####

print("\n### testing methods to convert list of tokens to pytorch tensor ###")
## this cell is to test the methods needed to convert the list of tokens
## in separate pytorch tensors into one big pytorch tensor with num of
## elements in each row as max_len


# tokenize all the text by doing it every few elements
# below two lines are for testing
test_list = [torch.tensor([[1]]),torch.tensor([[2,3]]),torch.tensor([[4, 5, 6]]),torch.tensor([[7,8,9,10]]),torch.tensor([[11,12,13,14,15]]),torch.tensor([[16, 17,18,19,20,21]]), torch.tensor([[22,23]])]


max_lensdf = 6
print(get_tokens_array_clean(test_list, max_length = max_lensdf, padding = True, padding_token = 100))

print(get_tokens_array_clean(test_list, max_length = max_lensdf, padding = False))
####

#### tokenizing for real now
print("keys: ", keys)

for key in keys:
  # after sanity check passes, tokenizing for real
  print(f"\n### tokenizing {key} set for real ###")

  # list of pt tensors of tokenized strings (step number at a time) in test dataset
  tokens_key_pt_list = []

  #for i in tqdm(range(0, len(clean_dataset[key]['text']), step), desc=f"tokenizing {key} dataset"):
  #  tokens_key_pt_list.append(tokenizer.encode(''.join(clean_dataset[key]['text'][i:i+step]), return_tensors='pt'))
  
  print("\nattempting multiprocessing tokenization (mainly helpful for train dataset)")
  # Process in chunks
  tokens_key_pt_list = process_in_chunks(clean_dataset[key]['text'], key, tokenizer, return_tensors, chunk_size = 1000)

  print("tokens_key_pt_list[0]: ", tokens_key_pt_list[0])
  print("len(tokens_key_pt_list): ", len(tokens_key_pt_list))


  print("\n### getting final token tensor for key data ###")

  ## getting final token tensor for key data; note that these are the input_ids; 
  ## the attention_masks is a tensor of the same shape but all 1s (meaning all the data is real data, none of it is a padding token)
  key_tokens_clean = get_tokens_array_clean(tokens_key_pt_list, max_length = max_len, padding = pad_tokens_bool)
  
  print(key_tokens_clean.size())
  print(f"total of {key_tokens_clean.size(0) * key_tokens_clean.size(1)} tokens for test")

  print("\n### Create the directory to save if it does not exist ###")
  # Create the directory if it does not exist
  os.makedirs(clean_token_dir, exist_ok=True)

  clean_key_tokens_filename = clean_token_dir + '/' + key + '_tokens_clean.pt'

  print("\n### saving key_tokens_clean ###")
  torch.save(key_tokens_clean, clean_key_tokens_filename)
  print("finished saving key clean tokens")

  check_saved_tokens_match(clean_key_tokens_filename, key, key_tokens_clean)



### ignore below, as of Feb 11, 2024


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




