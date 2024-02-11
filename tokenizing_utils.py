from itertools import chain


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
