import re
import os
import sys
from tqdm import tqdm
from typing import List
from datasets import load_dataset, Dataset


# filtering out empty texts
def filter_empty_texts(examples):
    return [s for s in examples if s]

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

# check for emails in datasets, if true, sys.exit(); else can remove all @ in dataset in a different method
def any_emails(dataset) -> bool:
    print("\n checking for email addresses now")
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    email_addresses = {}
    # Find all occurrences of the email pattern
    for i in tqdm(range(len(dataset)), desc ="checking for emails in dataset"):
        potential_emails = re.findall(email_pattern, dataset[i])
        if potential_emails:
            email_addresses[i] = potential_emails

    # Check if any email addresses were found
    if email_addresses:
        print("Found email addresses:", email_addresses)
        sys.exit()
    else:
        print("No email addresses found.")
    return False

# remove characters specified in the str chars_to_remove from text, returns cleaned text
def remove_chars(chars_to_remove: str, text: str):

    # Create a translation table mapping each character to None
    translation_table = str.maketrans('', '', chars_to_remove)

    # Remove the characters from the string
    cleaned_text = text.translate(translation_table)
    #print("\nthis is cleaned_text chars removed: ", cleaned_text)
    return cleaned_text

# removes whitespace inside " ", e.g. " hi " becomes "hi" in the text, returns cleaned text 
def clean_quoted_whitespace(text):
    # This pattern looks for a space after an opening quote or before a closing quote
    pattern = r'" *([^"]*?) *"'
    # This function will be used to replace the matched text
    def replacer(match):
        # This will replace the matched text with the text without leading/trailing spaces
        return '"' + match.group(1) + '"'
    # Replace all occurrences of the pattern with the result of the replacer function
    cleaned = re.sub(pattern, replacer, text)
    # Strip leading and trailing whitespace
    return cleaned.strip()

# removes whitespace in front of or behind specified strings, returns cleaned text
def remove_custom(text: str, lst_to_remove_comp: List, lst_to_strip_pre_ws: List, lst_to_strip_post_ws: List) -> str:
    for s in lst_to_remove_comp:
        text = text.replace(s, "")

    for s in lst_to_strip_pre_ws:
        text = text.replace(" "+ s, s) 
        
    for s in lst_to_strip_post_ws:
        text = text.replace(s+" ", s) 
        
    return text



# cleans each string in given list of strings by removing specified chars, and string for which you want to remove preceding or post white space
def clean_completely(dataset_list, chars_to_remove, lst_to_remove_comp, lst_to_strip_pre_ws, lst_to_strip_post_ws):
    print("\nin clean_completely")
    cleaned_lst = []
    
    print("removing empty strings")
    dataset_list = filter_empty_texts(dataset_list)
    print("finished removing empty strings")
    
    print("starting to clean each string in list")
    for string in dataset_list:
        text = remove_chars(chars_to_remove, string.strip())
        text = clean_quoted_whitespace(text)
        text = remove_custom(text, lst_to_remove_comp, lst_to_strip_pre_ws, lst_to_strip_post_ws)
        
        cleaned_lst.append(text.strip())

    print("finished cleaning each string in list")
    return cleaned_lst




def save_clean_dataset(filename: str, cleaned_dataset: List, uni_delim ="\0") -> None:
    print("\n### saving to file ###")
    # Using a unique delimiter to join and save strings
    # Here we use uni_delim as it's unlikely to be in the text
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(uni_delim.join(cleaned_dataset))
    print("\n### finished saving to file ###")


def load_clean_dataset(filename: str, uni_delim = "\0") -> List:
    print("\n### loading from file ###")
    # Load the strings back, splitting by the unique delimiter
    with open(filename, 'r', encoding='utf-8') as file:
        loaded_filtered_text = file.read().split(uni_delim)

    print("\n### finished loading from file ###")
    return loaded_filtered_text

