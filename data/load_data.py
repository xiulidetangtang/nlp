import transformers
import torch
from torch.utils.data import DataLoader, TensorDataset
import csv
from utils.utils import *
from data.preprocess import preprocess_text

def get_tokens(path, tokenizer : transformers.AutoTokenizer):
    """
    :return: the maximum number of tokens in the given csv training file, and all tokenized tokens.
    """
    tokens_list = []
    max_len = 0
    with open(path, 'r+') as f:
        reader = csv.reader(f)

        for row in reader:
            if row[0] == 'text_id': #ignore the first row
                continue
            
            tokens = tokenizer.tokenize(preprocess_text(row[1]))
            length = len(tokens)
            tokens_list.append(tokens)

            if length > max_len:
                max_len = length
        
        return max_len, tokens_list

def createDataLoader(device, path = None, pretrained = None, 
                     valid_size = 64,
                     batch_size = 16,):
    if path == None:
        path = './data/train.csv'
    if pretrained == None:
        pretrained = 'microsoft/deberta-v3-base'
        #pretrained = 'DebertaV2Tokenizer'

    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained,use_fast=False)

    scores = []
    print("Tokenize text...")
    max_len, tokens = get_tokens(path, tokenizer)
    print(f"Have a maximum sequence length of {max_len}...")
    #convert tokens to equally lengthed ids:
    input_ids = []
    input_masks = []
    for token in tokens:
        ids = tokenizer.convert_tokens_to_ids(token)
        paddings = max_len - len(ids)
        input_mask = [1]*len(ids) + [0]*paddings
        ids += [0]*paddings

        input_ids.append(ids)
        input_masks.append(input_mask)

    with open(path, 'r+') as f:
        reader = csv.reader(f)

        for row in reader:
            if row[0] == 'text_id': #ignore the first row
                continue
            
            scores.append(as_integer(row[2:]))

    input_ids = torch.tensor(input_ids, dtype=torch.int)
    input_masks=torch.tensor(input_masks, dtype=torch.int)
    scores    = torch.tensor(scores)

    #random shuffle
    idx = torch.randperm(input_ids.size(0))
    input_ids = input_ids[idx]
    scores    = scores[idx]
    input_masks=input_masks[idx]

    #input_ids = input_ids.to(device)
    #scores = scores.to(device)
    #input_masks = input_masks.to(device)

    #seperate into train set and valid set
    train_size = input_ids.size(0) - valid_size
    train_x = input_ids[:train_size]
    train_y = scores[:train_size]
    train_mask = input_masks[:train_size]

    valid_x = input_ids[train_size:]
    valid_y = scores[train_size:]
    valid_mask = input_masks[train_size:]

    dataset = TensorDataset(train_x, train_mask, train_y)
    train_loader = DataLoader(dataset, 
                              batch_size = batch_size,
                              shuffle=True)
    dataset_valid = TensorDataset(valid_x, valid_mask, valid_y)
    valid_loader = DataLoader(dataset_valid, 
                              batch_size = valid_size,
                              shuffle=True)

    return train_loader, valid_loader

def createEmbeddingDataset(device, dataloader, pretrained = None):
    if pretrained is None:
        pretrained = 'microsoft/deberta-v3-base'
    config= transformers.AutoConfig.from_pretrained(pretrained)
    model = transformers.AutoModel.from_pretrained(pretrained, config = config)
    
    model.eval()
    model = model.to(device)
    embed = None
    value = None
    with torch.no_grad():
        for count, (x, mask,  y) in enumerate(dataloader):
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)
            embedding = model(x, mask).last_hidden_state
            if embed is None:
                embed = embedding.cpu()
                value = y.cpu()
            else:
                embed = torch.cat((embed, embedding.cpu()), dim=0)
                value = torch.cat((value, y.cpu()), dim=0)

            if (count + 1) % 10 == 0:
                print(f"{count + 1}/{len(dataloader)} completed...")

        print(f"the total size is Embedding:{embed.size()}, Value:{value.size()}")
        dataset = TensorDataset(embed, value)

        return dataset
        
