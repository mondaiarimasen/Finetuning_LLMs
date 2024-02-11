from cleaning_data_utils import *


#### variables
keys = ["test","validation", "train"]
uni_delim = "\0"

wandb_run_name = "training_loop"
wandb_project = "gpt-2-finetuning"

uni_delim = "\0" # a unique delimiter to save and load filtered text from dataset
# Here we use "\0" (null character) as it's unlikely to be in the text

dataset_name = 'wikitext'
dataset_config = 'wikitext-103-v1'
dataset_path_head = os.getcwd()
home_dir = os.getcwd()
dataset_filename = '/wikitext-103-raw-v1'





#### loading saved cleaned tokens for test, train, validation sets

dataset = load_or_fetch_wikitext(dataset_name, dataset_config, dataset_path_head + dataset_filename)

print(dataset)


print(dataset['train'][:5])

print(dataset['train'])

# viewing first few elements
print("dataset['test']['text'][:5]: ", dataset['test']['text'][:5])
# 

chars_to_remove = ""
if any_emails(dataset['validation']['text']) == False:
    # now removing selected chars
    # Define the characters to be removed
    chars_to_remove = "@="


lst_to_remove_comp = [" / <unk> /" ," <unk> ", " <unk>", "<unk> "]
lst_to_strip_pre_ws = [":", ";", ",", ".", "- ", "'", ")", "!", "]"]
lst_to_strip_post_ws = ["-", "...", "(", "["]


clean_dataset = {}

for key in keys:
    dataset_to_clean = dataset[key]['text']#[15:45]
    clean_dataset[key] = {}
    print(f"\ncleaning {key} dataset")

    clean_dataset[key]['text'] = clean_completely(dataset_to_clean, chars_to_remove, lst_to_remove_comp, lst_to_strip_pre_ws, lst_to_strip_post_ws)
    print(f"this is last ten ele of cleaned_dataset[{key}] after clean_completely:")
    for item in clean_dataset[key]['text'][-20:]:
        print(item)

    #print("\n now printing original")
    #for item in dataset_to_clean:
    #    print(item)


#print(clean_completely(["2000 - 2005"], chars_to_remove, lst_to_remove_comp, lst_to_strip_pre_ws, lst_to_strip_post_ws))
#print("possible special character:", clean_completely(["2000 – 2005"], chars_to_remove, lst_to_remove_comp, lst_to_strip_pre_ws, lst_to_strip_post_ws))












#### saving filtered text

print("\n### Create the directory to save if it does not exist ###")
# Create the directory if it does not exist
directory = home_dir + "/" + dataset_config + "-filtered-text-lists"
os.makedirs(directory, exist_ok=True)

for key in keys:
    print("key: ", key)
    filename = "/" + key + "_text_filtered.txt"

    save_clean_dataset(directory + filename, clean_dataset[key]['text'], uni_delim = uni_delim)


    # loading filtered text to check integrity of saved file
    # Load the strings back, splitting by the unique delimiter
    loaded_filtered_text = load_clean_dataset(directory+filename,uni_delim = uni_delim)

    print("loaded_filtered_text[:5]: ", loaded_filtered_text[:5])

    # checking if loaded filtered text matches filtered text computed
    if loaded_filtered_text == clean_dataset[key]['text']:
        print("The lists are the same.")
    else:
        print("The lists are different.")



