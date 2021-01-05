import json
import numpy as np
from nltk_utils import tokenize,extract_stop_words,stem,bag_of_words
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



with open('intents.json',encoding='UTF-8') as f:
    intents = json.load(f)

all_words = list()
tags = list()
pattern_tag_data = list()

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        words = tokenize(pattern)
        words_without_stops = extract_stop_words(words)
        all_words.extend(words_without_stops)
        pattern_tag_data.append((words_without_stops,tag))

ignore_words = ['?','!','.',',',';','#']

all_words = [stem(w) for w in all_words if w not in ignore_words]

all_words = sorted(set(all_words))

tags = sorted(set(tags))

x_train = list()
y_train = list()

for (pattern_sentece,tag) in pattern_tag_data:
    bag = bag_of_words(pattern_sentece,all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)


x_train = np.array(x_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.number_of_samples = len(x_train)
        self.x_train = x_train
        self.y_train = y_train

    def __getitem__(self,index):
        return self.x_train[index], self.y_train[index]
    
    def __len__(self):
        return self.number_of_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset= dataset, batch_size=8, shuffle=True, num_workers = 0)