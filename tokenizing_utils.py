from itertools import chain
from typing import List
import torch.nn.functional as func
import torch
from torch import Tensor
from tqdm import tqdm
import sys
import multiprocessing as mp
from common_utils import *


def sanity_check_tokenize_idpt_splice(clean_dataset, tokenizer, step = 10):
    print("\n### conducting sanity check for tokenizing ###")
    # Sanity check: testing tokenizer on two subsets of train data but in one
    # case on the entire subset and in another case after splitting subset into two halves

    # entire subset
    tokens_encode_1 = tokenizer.encode(''.join(clean_dataset[:step]), return_tensors='pt')
    print(tokens_encode_1)
    print(len(tokens_encode_1[0]))

    # tokenizing the first step/2 strings then the next step/2 strings and comparing
    tokens_encode_2a = tokenizer.encode(''.join(clean_dataset[:step//2]), return_tensors='pt')
    print(tokens_encode_2a)
    print(len(tokens_encode_2a[0]))
    tokens_encode_2b = tokenizer.encode(''.join(clean_dataset[step//2:step]),return_tensors='pt')
    print(len(tokens_encode_2b[0]))


    # should print True for the sanity check to pass
    sanity_check_bool = all(a == b for a, b in zip(chain(tokens_encode_2a[0], tokens_encode_2b[0]), tokens_encode_1[0]))
    print("sanity_check_bool: ", sanity_check_bool)
    if sanity_check_bool == True:
        print("passed sanity check")
    else: 
        print("didn't pass sanity check")
        sys.exit()


# constructing the m * max_len 2d array of tokens from 1d tensor l1; m is ceiling of len of
# {string of all text} divided by max_len; we pad the tail
# returns reshaped 2d tensor and remaining part as 2d tensor with dim 1 * num of remaining ele
def separate_and_reshape(l1: torch.Tensor, max_len: int):
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


# helper method needed for multiprocessing in tokenization
def tokenize_data(data_chunk, tokenizer, return_tensors):
    # using the GPT2 tokenizer here!
    return tokenizer.encode(''.join(data_chunk), return_tensors=return_tensors)

# method to process tokenization in chunks, for multiprocessing
def process_in_chunks(dataset, key:str, tokenizer, return_tensors, chunk_size=10000):
    pool = mp.Pool(mp.cpu_count())
    results = []
    print("chunk_size: ", chunk_size)

    for i in tqdm(range(0, len(dataset), chunk_size), desc=f"tokenizing {key} dataset, mp"):
        chunk = dataset[i:i + chunk_size]
        result = pool.apply_async(tokenize_data, [chunk,tokenizer, return_tensors])
        results.append(result)

    pool.close()
    pool.join()

    print("type(result.get()): ", type(results[0].get()))
    return [result.get() for result in tqdm(results, desc = "getting results")]


# helper function to check if tokenization saved without a problem
def check_saved_tokens_match(clean_key_tokens_filename: str, key: str, key_tokens_clean: torch.Tensor):
    print("\n### checking if saved correctly ###")
    # checking if saved correctly
    loaded_tensor = load_clean_tokens(clean_key_tokens_filename, key)
    print(loaded_tensor)

    are_same = torch.equal(key_tokens_clean, loaded_tensor)
    print("loaded tensor and computed tensor are same: ", are_same)  # Output: True
    if are_same == False:
        sys.exit()

    # Check if the file exists
    if os.path.exists(clean_key_tokens_filename):
        print("File exists.")
    else:
        print("File does not exist.")
        sys.exit()


