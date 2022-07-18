import yaml
import pandas as pd
import torch
import gc
from torch.utils.data import DataLoader
import torch.nn.functional as F
from keras_preprocessing.sequence import pad_sequences


def load_config(config_file, default_config_dict=dict()):
    config_dict = {}
    with open(config_file, 'r') as f:
        conf = yaml.safe_load(f)

        for key, default in default_config_dict.items():
            if key in conf:
                config_dict[key] = conf[key]
            else:
                config_dict[key] = default

    return config_dict


def encode(tokenizer, text, MAX_LEN=512):
    input_ids = tokenizer.encode(
        text, padding=True, truncation=True, max_length=MAX_LEN)
    attention_mask = torch.tensor(
        [1] * len(input_ids) + [0] * (MAX_LEN - len(input_ids)))

    input_ids = pad_sequences(
        [input_ids], maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    input_ids = torch.tensor(input_ids).squeeze()

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }


def load_dataset(task, path, num_classes, tokenizer, batch_size, aug=True, multi_tokenizer=False):
    df = pd.read_csv(path)

    X = df['comments'].to_list()
    y = df['label'].to_list()

    if task == 'gender_bias':  # Data augmentation
        X_True = df['comments'][df['label'] == True].to_list()
        X_False = df['comments'][df['label'] == False].to_list()

        X_True_num = len(X_True)
        X_False_num = len(X_False)

        if aug:
            X_True *= (X_False_num // X_True_num)

        X = X_True + X_False
        y = [1] * len(X_True) + [0] * len(X_False)
    else:
        y_map = {
            'none': 0,
            'hate': 1,
            'offensive': 2,
        } if task == 'hate_speech' else {
            'none': 0,
            'others': 1,
            'gender': 2,
        }

        y = [y_map[y] for y in y]

    inputs, masks, labels = [], [], []
    for x, y in zip(X, y):
        if multi_tokenizer:
            inputs_ = []
            masks_ = []
            for tok in tokenizer:
                encoded = encode(tok, x)
                inputs_.append(encoded['input_ids'])
                masks_.append(encoded['attention_mask'])
            inputs.append(inputs_)
            masks.append(masks_)
            labels.append(F.one_hot(torch.Tensor(
                [y]).to(torch.int64), num_classes=num_classes)[0])
        else:
            encoded = encode(tokenizer, x)
            inputs.append(encoded['input_ids'])
            masks.append(encoded['attention_mask'])
            labels.append(F.one_hot(torch.Tensor(
                [y]).to(torch.int64), num_classes=num_classes)[0])

    dataset = list(zip(inputs, masks, labels))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def empty_mem():
    gc.collect()
    torch.cuda.empty_cache()


def freeze_bert(model, num_layers):
    for param in model.parameters():
        param.requires_grad = True

    n = 0
    for name, child in model.named_children():
        if n == 0:
            h = 0
            for param in child.parameters():
                if h <= num_layers:
                    param.requires_grad = False
                h += 1
        n += 1
